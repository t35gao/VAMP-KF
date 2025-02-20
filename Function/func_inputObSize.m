function [Ny] = func_inputObSize(Nx)
ob_type = input("Choose the observation matrix type:\n" + ...
                    "1 for Ny = Nx / 2 (underdetermined);\n" + ...
                    "2 for Ny = Nx * 2 (overdetermined):\n");
switch ob_type
    case 1;     Ny = Nx/2;
    case 2;     Ny = Nx*2;
    otherwise;  error('Invalid input: choose either 1 or 2')
end
end

