function [SNR] = func_inputSNR()
SNR_type = input("Choose the type of SNR:\n" + ...
                    "1 for SNR = 5 dB;\n" + ...
                    "2 for SNR = 20 dB:\n");
switch SNR_type
    case 1;     SNR = 5;
    case 2;     SNR = 20;
    otherwise;  error("Invalid input: choose either 1 or 2");
end
end

