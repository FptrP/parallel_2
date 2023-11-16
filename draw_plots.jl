import YAML
using GLMakie

function target_function(L::Float64, x::Float64, y::Float64, z::Float64, t::Float64)::Float64
  v1 = sin(2.0 * pi/L * x)
  v2 = sin(pi/L * y + pi)
  v3 = sin(2.0 * pi/L * z + 2.0 * pi)
    
  at = pi/3.0 * sqrt(4.0/(L^2) + 1.0/(L^2) + 4.0/(L^2)) 
  v4 = cos(at * t + pi)
  
  return v1 * v2 * v3 * v4
end

L::Float64 = 1.0
T::Float64 = 0.3
time_steps::Int32 = 1000
dim::Int32 = 128
a2::Float64 = 1.0/9.0
h::Float64 = L/(dim - 1)
#tau::Float64 = T/(time_steps - 1) 
tau::Float64 = h/10.0
T = (time_steps - 1) * tau

points = [((x - 1) * h, (y - 1) * h, (z - 1) * h) 
  for z in 1:dim for y in 1:dim for x in 1:dim]

const PTS_STEP = 4

opt_points = reshape(points, (dim, dim, dim))[begin:PTS_STEP:end, begin:PTS_STEP:end, begin:PTS_STEP:end]
opt_points = reshape(opt_points, :) 

function get_reference(time_step::Int)::Vector{Float64}
  t::Float64 = tau * time_step

  fun = (p::Tuple{Float64, Float64, Float64}) -> target_function(L, p[1], p[2], p[3], t)

  return fun.(opt_points)
end

@inline function laplacian(grid::Array{Float64, 3}, x::Int32, y::Int32, z::Int32)
  res::Float64 = 0.0
  uc = grid[z, y, x]
  
  x1 = x == 1 ? (dim - 1) : (x - 1)
  x2 = x == dim ? 2 : (x + 1)

  z1 = z == 1 ? (dim - 1) : (z - 1)
  z2 = z == dim ? 2 : (z + 1)

  res += grid[z, y, x1] - 2.0 * uc + grid[z, y, x2]
  res += grid[z, y - 1, x] - 2.0 * uc + grid[z, y + 1, x]
  res += grid[z1, y, x] - 2.0 * uc + grid[z2, y, x]

  return res/(h^2)
end

function calculate_grid()
  out_grids = []

  grids = [zeros((dim, dim, dim)), zeros((dim, dim, dim)), zeros((dim, dim, dim))]
  
  #init

  for zi::Int32 in 1:dim
    for yi::Int32 in 1:dim
      for xi::Int32 in 1:dim
        grids[1][zi, yi, xi] = target_function(L, (xi - 1)*h, (yi-1)*h, (zi - 1)*h, 0.0)
      end
    end
  end 

  for zi::Int32 in 1:dim
    for yi::Int32 in 2:(dim - 1)
      for xi::Int32 in 1:dim
        grids[2][zi, yi, xi] = grids[1][zi, yi, xi] + a2 * 0.5 * (tau^2) * laplacian(grids[1], xi, yi, zi)
      end
    end
  end 
  
  push!(out_grids, copy(grids[1])[begin:PTS_STEP:end, begin:PTS_STEP:end, begin:PTS_STEP:end])
  push!(out_grids, copy(grids[2])[begin:PTS_STEP:end, begin:PTS_STEP:end, begin:PTS_STEP:end])

  for ti::Int32 in 2 : (time_steps - 1)
    println("Running step : $ti")

    g = grids[(ti % 3) + 1]
    g1 = grids[((ti - 1) % 3) + 1]
    g2 = grids[((ti - 2) % 3) + 1]
    
    for zi::Int32 in 1:dim
      for yi::Int32 in 2:(dim - 1)
        Threads.@threads for xi::Int32 in 1:dim
          un = g1[zi, yi, xi]
          un_1 = g2[zi, yi, xi]
          l = laplacian(g1, xi, yi, zi)
          g[zi, yi, xi] = (tau^2) * (a2 * l) + 2.0 * un - un_1
        end
      end
    end

    push!(out_grids, copy(g)[begin:PTS_STEP:end, begin:PTS_STEP:end, begin:PTS_STEP:end])
  end

  return out_grids
end

computed = calculate_grid()
reference = Array{Any}(undef, time_steps)
errors = Array{Any}(undef, time_steps)

Threads.@threads for i in eachindex(reference)
  reference[i] = get_reference((i - 1))
  
  #c = computed[i][begin:PTS_STEP:end, begin:PTS_STEP:end, begin:PTS_STEP:end]
  computed[i] = reshape(computed[i], :)
  errors[i] = abs.(reference[i] - computed[i])
  println("Error $i : ", maximum(errors[i]))
end

max_error::Float64 = 0.0
max_error = maximum(map(x -> maximum(x), errors))
println("Max error = $max_error")

const ALPHA = 0.1
const MARKERSIZE = 9

function record_grid(name, colors, range, value_name)
  plt = scatter(opt_points, color=zeros(length(opt_points)),
     alpha=ALPHA, markersize=MARKERSIZE, colorrange=range)
  
  l = Label(plt.figure[0, 0], "step=0")
  Colorbar(plt.figure[1, 1][1, 2], label=value_name, limits=range, vertical=true)

  iter = 1:length(colors)

  record(plt.figure, name, iter, framerate=30) do i
    plt.plot.color = colors[i]
    l.text = "step=$i"
  end

end

record_grid("anim/reference.mp4", reference, (-1, 1), "Аналитическое u")
record_grid("anim/computed.mp4", computed, (-1, 1), "Вычисленное u")
record_grid("anim/error.mp4", errors, (0, max_error), "Модуль ошибки")