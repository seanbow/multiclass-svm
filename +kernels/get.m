function K = get(type)

if strcmpi(type, "rbf") || strcmpi(type, "gaussian")
    K = @kernels.rbf;
elseif strcmpi(type, "polynomial")
    K = @kernels.polynomial;
else
    % fall back to linear, which requires no params
    K = @kernels.linear;
end

end

