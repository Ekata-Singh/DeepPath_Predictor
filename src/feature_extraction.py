#!/usr/bin/env python3
"""
Feature extraction module for RTL combinational depth prediction.
This module extracts relevant features from Verilog RTL code.
"""

import re
import os
import pyverilog.vparser.parser as parser
from pyverilog.vparser.ast import *
from collections import defaultdict

class RTLFeatureExtractor:
    """
    Class to extract features from RTL code that are relevant for
    predicting combinational depth.
    """
    
    def __init__(self, rtl_file):
        """
        Initialize the feature extractor with an RTL file.
        
        Args:
            rtl_file (str): Path to the RTL file
        """
        self.rtl_file = rtl_file
        self.ast = None
        self.signals = {}
        self.dependencies = defaultdict(list)
        self.parse_rtl()
        
    def parse_rtl(self):
        """Parse the RTL file using PyVerilog."""
        try:
            ast, _ = parser.parse([self.rtl_file])
            self.ast = ast
            self._extract_signals()
            self._extract_dependencies()
        except Exception as e:
            print(f"Error parsing RTL file: {e}")
            
    def _extract_signals(self):
        """Extract all signals from the RTL."""
        for module in self.ast.description.definitions:
            if type(module) is ModuleDef:
                # Extract ports
                for port in module.portlist.ports:
                    self.signals[port.name] = {
                        'type': 'port',
                        'direction': None,  # Will be updated later
                    }
                
                # Extract items (wires, regs, etc.)
                for item in module.items:
                    if isinstance(item, Decl):
                        for decl in item.list:
                            if isinstance(decl, Wire):
                                self.signals[decl.name] = {
                                    'type': 'wire',
                                    'width': self._get_width(decl.width),
                                }
                            elif isinstance(decl, Reg):
                                self.signals[decl.name] = {
                                    'type': 'reg',
                                    'width': self._get_width(decl.width),
                                }
                    elif isinstance(item, Input):
                        for inp in item.list:
                            self.signals[inp.name] = {
                                'type': 'input',
                                'width': self._get_width(item.width),
                            }
                    elif isinstance(item, Output):
                        for out in item.list:
                            self.signals[out.name] = {
                                'type': 'output',
                                'width': self._get_width(item.width),
                            }
                    elif isinstance(item, Inout):
                        for inout in item.list:
                            self.signals[inout.name] = {
                                'type': 'inout',
                                'width': self._get_width(item.width),
                            }
    
    def _get_width(self, width):
        """Extract the width of a signal."""
        if width is None:
            return 1
        try:
            if isinstance(width, Width):
                msb = width.msb.value
                lsb = width.lsb.value
                return int(msb) - int(lsb) + 1
        except:
            pass
        return 1
    
    def _extract_dependencies(self):
        """Extract signal dependencies from assignments."""
        for module in self.ast.description.definitions:
            if type(module) is ModuleDef:
                for item in module.items:
                    if isinstance(item, Assign):
                        lhs = self._get_signal_name(item.left)
                        rhs_signals = self._extract_rhs_signals(item.right)
                        self.dependencies[lhs].extend(rhs_signals)
                    elif isinstance(item, Always):
                        self._process_always_block(item)
    
    def _process_always_block(self, always):
        """Process an always block to extract dependencies."""
        if isinstance(always.statement, Block):
            for stmt in always.statement.statements:
                if isinstance(stmt, IfStatement):
                    self._process_if_statement(stmt)
                elif isinstance(stmt, NonblockingSubstitution) or isinstance(stmt, BlockingSubstitution):
                    lhs = self._get_signal_name(stmt.left)
                    rhs_signals = self._extract_rhs_signals(stmt.right)
                    self.dependencies[lhs].extend(rhs_signals)
                elif isinstance(stmt, Case):
                    self._process_case_statement(stmt)
    
    def _process_if_statement(self, if_stmt):
        """Process an if statement to extract dependencies."""
        cond_signals = self._extract_rhs_signals(if_stmt.cond)
        
        # Process true path
        if isinstance(if_stmt.true_statement, Block):
            for stmt in if_stmt.true_statement.statements:
                if isinstance(stmt, NonblockingSubstitution) or isinstance(stmt, BlockingSubstitution):
                    lhs = self._get_signal_name(stmt.left)
                    rhs_signals = self._extract_rhs_signals(stmt.right)
                    self.dependencies[lhs].extend(rhs_signals)
                    self.dependencies[lhs].extend(cond_signals)  # Condition affects the assignment
        
        # Process false path if it exists
        if if_stmt.false_statement:
            if isinstance(if_stmt.false_statement, Block):
                for stmt in if_stmt.false_statement.statements:
                    if isinstance(stmt, NonblockingSubstitution) or isinstance(stmt, BlockingSubstitution):
                        lhs = self._get_signal_name(stmt.left)
                        rhs_signals = self._extract_rhs_signals(stmt.right)
                        self.dependencies[lhs].extend(rhs_signals)
                        self.dependencies[lhs].extend(cond_signals)  # Condition affects the assignment
    
    def _process_case_statement(self, case):
        """Process a case statement to extract dependencies."""
        case_signals = self._extract_rhs_signals(case.comp)
        
        for case_item in case.caselist:
            if isinstance(case_item.statement, Block):
                for stmt in case_item.statement.statements:
                    if isinstance(stmt, NonblockingSubstitution) or isinstance(stmt, BlockingSubstitution):
                        lhs = self._get_signal_name(stmt.left)
                        rhs_signals = self._extract_rhs_signals(stmt.right)
                        self.dependencies[lhs].extend(rhs_signals)
                        self.dependencies[lhs].extend(case_signals)  # Case condition affects the assignment
    
    def _get_signal_name(self, node):
        """Extract the signal name from an AST node."""
        if isinstance(node, Identifier):
            return node.name
        elif isinstance(node, Partselect):
            return node.var.name
        elif isinstance(node, Pointer):
            return node.var.name
        return None
    
    def _extract_rhs_signals(self, node):
        """Extract all signals used in a right-hand side expression."""
        signals = []
        
        if node is None:
            return signals
        
        if isinstance(node, Identifier):
            signals.append(node.name)
        elif isinstance(node, Partselect) or isinstance(node, Pointer):
            signals.append(node.var.name)
        elif hasattr(node, 'left') and hasattr(node, 'right'):
            signals.extend(self._extract_rhs_signals(node.left))
            signals.extend(self._extract_rhs_signals(node.right))
        elif hasattr(node, 'args'):
            for arg in node.args:
                signals.extend(self._extract_rhs_signals(arg))
        
        return signals
    
    def extract_features(self, signal_name):
        """
        Extract features for a specific signal.
        
        Args:
            signal_name (str): Name of the signal to extract features for
            
        Returns:
            dict: Dictionary of features
        """
        if signal_name not in self.signals and signal_name not in self.dependencies:
            return None
        
        # Calculate fan-in (number of signals that directly affect this signal)
        fan_in = len(set(self.dependencies.get(signal_name, [])))
        
        # Count operators in the RTL for this signal
        operators_count = self._count_operators(signal_name)
        
        # Count conditional statements affecting this signal
        conditional_count = self._count_conditionals(signal_name)
        
        # Count different types of operations
        arithmetic_ops = self._count_arithmetic_ops(signal_name)
        logical_ops = self._count_logical_ops(signal_name)
        comparison_ops = self._count_comparison_ops(signal_name)
        
        # Count multiplexers
        mux_count = self._count_muxes(signal_name)
        
        return {
            'module_name': os.path.basename(self.rtl_file).split('.')[0],
            'signal_name': signal_name,
            'fan_in': fan_in,
            'operators_count': operators_count,
            'conditional_count': conditional_count,
            'arithmetic_ops': arithmetic_ops,
            'logical_ops': logical_ops,
            'comparison_ops': comparison_ops,
            'mux_count': mux_count
        }
    
    def _count_operators(self, signal_name):
        """Count the number of operators used in expressions that define the signal."""
        # This is a simplified implementation
        # In a real-world scenario, you would parse the AST more thoroughly
        count = 0
        for module in self.ast.description.definitions:
            if type(module) is ModuleDef:
                for item in module.items:
                    if isinstance(item, Assign) and self._get_signal_name(item.left) == signal_name:
                        count += self._count_operators_in_node(item.right)
        return max(1, count)  # At least 1 operator
    
    def _count_operators_in_node(self, node):
        """Recursively count operators in an AST node."""
        if node is None:
            return 0
        
        count = 0
        
        # Check if this node is an operator
        if isinstance(node, Operator):
            count += 1
        
        # Recursively check children
        if hasattr(node, 'left') and hasattr(node, 'right'):
            count += self._count_operators_in_node(node.left)
            count += self._count_operators_in_node(node.right)
        elif hasattr(node, 'args'):
            for arg in node.args:
                count += self._count_operators_in_node(arg)
        
        return count
    
    def _count_conditionals(self, signal_name):
        """Count conditional statements affecting the signal."""
        # Simplified implementation
        count = 0
        for module in self.ast.description.definitions:
            if type(module) is ModuleDef:
                for item in module.items:
                    if isinstance(item, Always):
                        count += self._count_conditionals_in_always(item, signal_name)
        return count
    
    def _count_conditionals_in_always(self, always, signal_name):
        """Count conditionals in an always block that affect the signal."""
        count = 0
        if isinstance(always.statement, Block):
            for stmt in always.statement.statements:
                if isinstance(stmt, IfStatement):
                    if self._if_affects_signal(stmt, signal_name):
                        count += 1
                elif isinstance(stmt, Case):
                    if self._case_affects_signal(stmt, signal_name):
                        count += 1
        return count
    
    def _if_affects_signal(self, if_stmt, signal_name):
        """Check if an if statement affects the signal."""
        # Check true path
        if isinstance(if_stmt.true_statement, Block):
            for stmt in if_stmt.true_statement.statements:
                if isinstance(stmt, (NonblockingSubstitution, BlockingSubstitution)):
                    if self._get_signal_name(stmt.left) == signal_name:
                        return True
        
        # Check false path
        if if_stmt.false_statement and isinstance(if_stmt.false_statement, Block):
            for stmt in if_stmt.false_statement.statements:
                if isinstance(stmt, (NonblockingSubstitution, BlockingSubstitution)):
                    if self._get_signal_name(stmt.left) == signal_name:
                        return True
        
        return False
    
    def _case_affects_signal(self, case, signal_name):
        """Check if a case statement affects the signal."""
        for case_item in case.caselist:
            if isinstance(case_item.statement, Block):
                for stmt in case_item.statement.statements:
                    if isinstance(stmt, (NonblockingSubstitution, BlockingSubstitution)):
                        if self._get_signal_name(stmt.left) == signal_name:
                            return True
        return False
    
    def _count_arithmetic_ops(self, signal_name):
        """Count arithmetic operations affecting the signal."""
        # Simplified implementation
        count = 0
        arithmetic_ops = ['+', '-', '*', '/', '%', '<<', '>>']
        
        for module in self.ast.description.definitions:
            if type(module) is ModuleDef:
                for item in module.items:
                    if isinstance(item, Assign) and self._get_signal_name(item.left) == signal_name:
                        count += self._count_specific_ops_in_node(item.right, arithmetic_ops)
        return count
    
    def _count_logical_ops(self, signal_name):
        """Count logical operations affecting the signal."""
        # Simplified implementation
        count = 0
        logical_ops = ['&', '|', '^', '~', '&&', '||', '!']
        
        for module in self.ast.description.definitions:
            if type(module) is ModuleDef:
                for item in module.items:
                    if isinstance(item, Assign) and self._get_signal_name(item.left) == signal_name:
                        count += self._count_specific_ops_in_node(item.right, logical_ops)
        return count
    
    def _count_comparison_ops(self, signal_name):
        """Count comparison operations affecting the signal."""
        # Simplified implementation
        count = 0
        comparison_ops = ['==', '!=', '<', '>', '<=', '>=']
        
        for module in self.ast.description.definitions:
            if type(module) is ModuleDef:
                for item in module.items:
                    if isinstance(item, Assign) and self._get_signal_name(item.left) == signal_name:
                        count += self._count_specific_ops_in_node(item.right, comparison_ops)
        return count
    
    def _count_specific_ops_in_node(self, node, op_list):
        """Count specific operators in an AST node."""
        if node is None:
            return 0
        
        count = 0
        
        # Check if this node is one of the specified operators
        if isinstance(node, Operator) and node.__class__.__name__ in op_list:
            count += 1
        
        # Recursively check children
        if hasattr(node, 'left') and hasattr(node, 'right'):
            count += self._count_specific_ops_in_node(node.left, op_list)
            count += self._count_specific_ops_in_node(node.right, op_list)
        elif hasattr(node, 'args'):
            for arg in node.args:
                count += self._count_specific_ops_in_node(arg, op_list)
        
        return count
    
    def _count_muxes(self, signal_name):
        """Count multiplexers (conditional operators) affecting the signal."""
        # Simplified implementation - count ternary operators and case statements
        count = 0
        
        for module in self.ast.description.definitions:
            if type(module) is ModuleDef:
                for item in module.items:
                    if isinstance(item, Assign) and self._get_signal_name(item.left) == signal_name:
                        count += self._count_ternary_ops(item.right)
                    elif isinstance(item, Always):
                        if isinstance(item.statement, Block):
                            for stmt in item.statement.statements:
                                if isinstance(stmt, Case) and self._case_affects_signal(stmt, signal_name):
                                    count += len(stmt.caselist)
        
        return count
    
    def _count_ternary_ops(self, node):
        """Count ternary operators in an AST node."""
        if node is None:
            return 0
        
        count = 0
        
        # Check if this node is a ternary operator
        if isinstance(node, Cond):
            count += 1
        
        # Recursively check children
        if hasattr(node, 'cond') and hasattr(node, 'true_value') and hasattr(node, 'false_value'):
            count += self._count_ternary_ops(node.cond)
            count += self._count_ternary_ops(node.true_value)
            count += self._count_ternary_ops(node.false_value)
        elif hasattr(node, 'left') and hasattr(node, 'right'):
            count += self._count_ternary_ops(node.left)
            count += self._count_ternary_ops(node.right)
        elif hasattr(node, 'args'):
            for arg in node.args:
                count += self._count_ternary_ops(arg)
        
        return count

# Simple function to extract features from an RTL file for a specific signal
def extract_features_from_rtl(rtl_file, signal_name):
    """
    Extract features from an RTL file for a specific signal.
    
    Args:
        rtl_file (str): Path to the RTL file
        signal_name (str): Name of the signal to extract features for
        
    Returns:
        dict: Dictionary of features or None if extraction fails
    """
    try:
        extractor = RTLFeatureExtractor(rtl_file)
        return extractor.extract_features(signal_name)
    except Exception as e:
        import shutil
        if "No such file or directory: 'iverilog'" in str(e) or shutil.which('iverilog') is None:
            print(f"Error parsing RTL file: Icarus Verilog (iverilog) is not installed.")
            print("For full RTL parsing functionality, please install Icarus Verilog.")
        else:
            print(f"Error parsing RTL file: {str(e)}")
        return None

# Function to extract features for all signals in an RTL file
def extract_all_features(rtl_file):
    """
    Extract features for all signals in an RTL file.
    
    Args:
        rtl_file (str): Path to the RTL file
        
    Returns:
        list: List of feature dictionaries for all signals
    """
    extractor = RTLFeatureExtractor(rtl_file)
    features = []
    
    for signal_name in extractor.signals.keys():
        signal_features = extractor.extract_features(signal_name)
        if signal_features:
            features.append(signal_features)
    
    return features

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python feature_extraction.py <rtl_file> <signal_name>")
        sys.exit(1)
    
    rtl_file = sys.argv[1]
    signal_name = sys.argv[2]
    
    features = extract_features_from_rtl(rtl_file, signal_name)
    if features:
        print(f"Features for signal {signal_name} in {rtl_file}:")
        for key, value in features.items():
            print(f"  {key}: {value}")
    else:
        print(f"Signal {signal_name} not found in {rtl_file}") 