module alu (
    input wire clk,
    input wire reset,
    input wire [3:0] opcode,
    input wire [15:0] operand_a,
    input wire [15:0] operand_b,
    output reg [15:0] result,
    output wire overflow_flag
);

    // Intermediate signals
    wire [15:0] add_result, sub_result, and_result, or_result, xor_result;
    wire [15:0] shift_left, shift_right;
    wire [15:0] mult_result;
    wire carry_out, borrow;
    wire overflow_add, overflow_sub, overflow_mult;
    
    // Level 1 operations
    assign add_result = operand_a + operand_b;
    assign sub_result = operand_a - operand_b;
    assign and_result = operand_a & operand_b;
    assign or_result = operand_a | operand_b;
    assign xor_result = operand_a ^ operand_b;
    
    // Level 2 operations
    assign shift_left = operand_a << operand_b[3:0];
    assign shift_right = operand_a >> operand_b[3:0];
    assign mult_result = operand_a * operand_b;
    
    // Level 3 - Carry and borrow detection
    assign carry_out = (operand_a + operand_b) > 16'hFFFF;
    assign borrow = operand_a < operand_b;
    
    // Level 4 - Overflow detection
    assign overflow_add = (operand_a[15] == operand_b[15]) && (add_result[15] != operand_a[15]);
    assign overflow_sub = (operand_a[15] != operand_b[15]) && (sub_result[15] != operand_a[15]);
    assign overflow_mult = (mult_result[15:0] != mult_result);
    
    // Level 5 - Final overflow flag (depth = 6)
    assign overflow_flag = (opcode == 4'b0000) ? overflow_add :
                          (opcode == 4'b0001) ? overflow_sub :
                          (opcode == 4'b0010) ? overflow_mult :
                          (opcode == 4'b0011) ? carry_out :
                          (opcode == 4'b0100) ? borrow : 1'b0;
    
    // Register the result
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            result <= 16'h0000;
        end else begin
            case (opcode)
                4'b0000: result <= add_result;
                4'b0001: result <= sub_result;
                4'b0010: result <= mult_result[15:0];
                4'b0011: result <= and_result;
                4'b0100: result <= or_result;
                4'b0101: result <= xor_result;
                4'b0110: result <= shift_left;
                4'b0111: result <= shift_right;
                default: result <= 16'h0000;
            endcase
        end
    end

endmodule 