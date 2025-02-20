function [Mat] = func_randStdN(size, flag_RC)
% Returns a matrix of standard real (or complex) normal distributed random numbers.
%   @size:      [m,n] corresponds to a m-by-n matrix
%   @flag_RC:   flag: 'R' for real and 'C' for complex
%
%   @Mat:       returned matrix

switch flag_RC
    case 'R';   Mat = randn(size);
    case 'C';   Mat = sqrt(1/2)*(randn(size)+1i*randn(size));
    otherwise;  error("Argument flag_RC must be either 'R' or 'C'.");
end
end

