# encoding rule: an integer
# idx = 1, 2, 3

function Chained_CB3_II(n, x)
    vec = zeros(Float64, 3)
    for i = 1:(n-1)
        vec[1] += x[i]^4 + x[i+1]^2
        vec[2] += (2 - x[i])^2 + (2 - x[i+1])^2
        vec[3] += 2 * exp(- x[i] + x[i+1])
    end
    res = maximum(vec)
    code = argmax(vec)
    return code, res 
end

# the component function of Chained_CB3_II
function Chained_CB3_II_cpt(n, code, x)
    res = 0
    for i = 1:(n-1)
        if code == 1
            res += x[i]^4 + x[i+1]^2
        elseif code == 2
            res += (2 - x[i])^2 + (2 - x[i+1])^2
        else
            res += 2 * exp(- x[i] + x[i+1])
        end
    end
    return res 
end

function Chained_CB3_II_init(n)
    x_init = zeros(Float64, n)
    for i = 1:n 
        x_init[i] = 2
    end
    return x_init
end