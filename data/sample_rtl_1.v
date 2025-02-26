module counter (
    input wire clk,
    input wire reset,
    input wire enable,
    output reg [7:0] count,
    output wire max_reached
);

    wire intermediate1, intermediate2, intermediate3;
    
    // Some combinational logic with depth = 3
    assign intermediate1 = count[0] & count[1] | count[2];
    assign intermediate2 = count[3] ^ count[4] & count[5];
    assign intermediate3 = intermediate1 | (intermediate2 & count[6]);
    
    // Output with depth = 4
    assign max_reached = intermediate3 & count[7] | (count == 8'hFF);
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            count <= 8'h00;
        end else if (enable) begin
            count <= count + 1;
        end
    end

endmodule 