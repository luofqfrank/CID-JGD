# encoding rules: n tuple
# (sgn1, sgn2, ..., sgn_n): sgn_i for |x_i|

function brown_func2(n, x)
    code = zeros(Int, n)
    for i = 1:n 
        if x[i] >= 0
            code[i] = 1
        else
            code[i] = -1
        end
    end
    res = 0
    for i = 1:(n-1)
        res += (code[i] * x[i])^(x[i+1]^2+1) + (code[i+1] * x[i+1])^(x[i]^2+1)
    end
    return code, res 
end


function brown_func2_cpt(n, code, x)
    res = 0
    for i = 1:(n-1)
        res += (code[i] * x[i])^(x[i+1]^2 + 1) + (code[i+1] * x[i+1])^(x[i]^2 + 1)
    end
    return res 
end

function brown_func2_init(n)
    x_init = zeros(Float64, n) 
    for i = 1:n 
        if i % 2 == 1 
            x_init[i] = -1 
        else 
            x_init[i] = 1
        end
    end
    return x_init
end

