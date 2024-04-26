# encoding rules: integer
# idx: 1, 2 indicating whether the first or second component in max is active 

function Chained_Crescent_I(n, x)
    f1, f2, res = 0, 0, 0
    idx = 1
    for i = 1:(n-1)
        f1 += x[i]^2 + (x[i+1]-1)^2 + (x[i+1]-1)
        f2 += - x[i]^2 - (x[i+1]-1)^2 + (x[i+1]+1)
    end
    if f1 >= f2 
        res = f1 
    else 
        res = f2 
        idx = 2
    end
    return idx, res  
end


function Chained_Crescent_I_cpt(n, idx, x)
    res = 0
    for i = 1:(n-1)
        if idx == 1
            res += x[i]^2 + (x[i+1]-1)^2 + (x[i+1]-1)
        else 
            res += - x[i]^2 - (x[i+1]-1)^2 + (x[i+1]+1)
        end
    end
    return res
end 

function Chained_Crescent_I_init(n)
    x_init = zeros(Float64, n)
    for i = 1:n 
        if i % 2 == 1 
            x_init[i] = - 1.5
        else 
            x_init[i] = 2.0
        end
    end
    return x_init
end