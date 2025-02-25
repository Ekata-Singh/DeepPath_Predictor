## DeepPath_Predictor
This project develops an AI-powered model to predict the combinational complexity and depth of signals in a circuit, enabling early detection of timing violations before running full Static Timing Analysis (STA). 

# Problem Statement
Timing analysis is a crucial step in the design of any complex IP/SoC. However, timing analysis reports are generated after synthesis is complete, which is a very time consuming process. This leads to overall delays in the project execution time as timing violations can require architectural refactoring.

Creating an AI algorithm to predict combinational logic depth of signals in behavioural RTL can greatly speed up this process.

# Approach
**Dataset Creation:** Collected a dataset of RTL (Register Transfer Level) modules, capturing various signals along with their actual combinational depths.
**Feature Engineering:** Extracted key features from the RTL code that impact combinational depth, including:
      1. **Fan-in**: Number of signals directly influencing the target signal.
      2. **Fan-out**: Number of signals directly driven by the target signal.
      3. **Signal Depth:** Maximum depth of a signal in the combinational logic path.
      4. **Average Signal Path Length:** Average number of logic gates between inputs and outputs.
      5. **Number of Signal Assignments:** Measures how frequently signals are reassigned in RTL code.
      6. **Operator Count:** Total number of operators used in expressions.
      7. **Conditional Count:** Number of conditional statements.
      8. **Arithmetic Operations Count:** Instances of arithmetic computations.
      9. **Logical Operations Count:** Instances of logical computations.
      10. **Comparison Operations Count:** Number of comparison operations.
      11. **Multiplexer Count:** Number of multiplexers used.
      12. **Critical Path Length:** Maximum combinational path length between registers.
      13. **Gate-Level Path Delay:** Estimated delay based on logic gates used.
      14. **Sequential Depth:** Number of flip-flop stages in a given RTL module.
      15. **Pipeline Stages:** Count of pipeline registers in the module.
**Model Selection:** Evaluated multiple machine learning algorithms to identify the most effective predictor.
**Training:** Trained the chosen model using the prepared dataset.
**Evaluation:** Assessed the model’s accuracy and overall performance to ensure reliable predictions.


