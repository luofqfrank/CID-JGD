# encoding rules: n-1 tuple
# (sgn1, sgn2, ..., sgn_{n-1}): sgn_i for |x^2_i + x^2_{i+1} - 1|

function Chained_Mifflin2(n, x)
    sgn = zeros(Int, n-1)
    res = 0
    for i = 1:(n-1)
        res += - x[i] + 2 * (x[i]^2 + x[i+1]^2 - 1) + 1.75 * abs(x[i]^2 + x[i+1]^2 - 1)
        if x[i]^2 + x[i+1]^2 - 1 >= 0
            sgn[i] = 1
        else
            sgn[i] = -1
        end
    end
    code = tuple(sgn...)
    return code, res 
end


function Chained_Mifflin2_cpt(n, code, x)
    res = 0
    for i = 1:(n-1)
        res += - x[i] + 2 * (x[i]^2 + x[i+1]^2 - 1) + 1.75 * code[i] * (x[i]^2 + x[i+1]^2 - 1)
    end
    return res 
end


function Chained_Mifflin2_init(n)
    x_init = zeros(Float64, n) 
    for i = 1:n 
        x_init[i] = -1
    end
    return x_init
end