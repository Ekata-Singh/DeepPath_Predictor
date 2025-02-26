module fifo_controller (
    input wire clk,
    input wire reset,
    input wire write_en,
    input wire read_en,
    input wire [7:0] data_in,
    output reg [7:0] data_out,
    output wire full,
    output wire empty,
    output wire almost_full,
    output wire almost_empty
);

    // Internal registers
    reg [3:0] write_ptr;
    reg [3:0] read_ptr;
    reg [4:0] count;
    reg [7:0] memory [0:15];
    
    // Simple status flags (depth = 1)
    assign empty = (count == 0);
    assign full = (count == 16);
    
    // Level 2 - Threshold detection
    wire half_full = (count >= 8);
    wire quarter_full = (count >= 4);
    wire three_quarters_full = (count >= 12);
    
    // Level 3 - Complex status flags
    assign almost_empty = !empty && (count <= 2);
    assign almost_full = !full && (count >= 14);
    
    // Level 4 - Warning signals
    wire overflow_warning = full && write_en;
    wire underflow_warning = empty && read_en;
    
    // Level 5 - Error detection
    wire error_condition = overflow_warning || underflow_warning;
    
    // Level 6 - Complex condition for debugging
    wire debug_trigger = error_condition && (half_full ^ quarter_full) && three_quarters_full;
    
    // Write operation
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            write_ptr <= 4'h0;
        end else if (write_en && !full) begin
            memory[write_ptr] <= data_in;
            write_ptr <= write_ptr + 1;
        end
    end
    
    // Read operation
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            read_ptr <= 4'h0;
            data_out <= 8'h00;
        end else if (read_en && !empty) begin
            data_out <= memory[read_ptr];
            read_ptr <= read_ptr + 1;
        end
    end
    
    // Count update
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            count <= 5'h0;
        end else begin
            case ({read_en && !empty, write_en && !full})
                2'b01: count <= count + 1;
                2'b10: count <= count - 1;
                2'b11: count <= count;
                2'b00: count <= count;
            endcase
        end
    end

endmodule 