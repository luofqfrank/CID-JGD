#using Pkg
#Pkg.add("ForwardDiff")
using ForwardDiff

# encoding rules (dictionary): i (the i^th component is active in max)
function gen_MAXQ(n, x)
  squared_x = x.^2
  val = maximum(squared_x)
  code = argmax(squared_x)
  return code, val
end

# get the specified component function
function gen_MAXQ_cpt(n, code, x)
  return x[code]^2
end

function gen_MAXQ_init(n)
  x_init = zeros(Float64, n)
  for i = 1:n 
    if i % 2 == 1
      x_init[i] = i 
    else
      x_init[i] = -i 
    end
  end
  return x_init 
end





