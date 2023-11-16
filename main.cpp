#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>

#include <omp.h>

struct TimeProfiler
{
  TimeProfiler(double &out)
  {
    measuredTime = &out;
    t_start = omp_get_wtime();
  }

  void stop()
  {
    if (!measuredTime)
      return;
    double t_end = omp_get_wtime();
    *measuredTime += 1000.0 * (t_end - t_start); 
    measuredTime = NULL;
  }

  ~TimeProfiler()
  {
    stop();
  }

  double *measuredTime;
  double t_start;
};

struct SolveParams
{
  // dimensions
  double Lx;
  double Ly;
  double Lz;
  //time
  double T;

  //a^2
  double a2;

  int num_threads;
  int dim_steps; // per Lx Ly Lz
  int time_steps; 

  void init()
  {
    if (!(Lx == Ly && Ly == Lz))
      throw "Invalid grid size";
    
    h = Lx/(dim_steps - 1);
    tau = T/(time_steps - 1);
  }

  double h;
  double tau;
};

struct RunStats
{
  SolveParams src;
  double init_time_ms;
  double iter_time_ms;
  double error;
};

struct PointGrid
{
  PointGrid() : dimX(0), dimY(0), dimZ(0) {}

  PointGrid(int dx, int dy, int dz)
  {
    init(dx, dy, dz);
  }

  void init(int dx, int dy, int dz)
  {
    dimX = dx; dimY = dy; dimZ = dz;
    points.resize(dimX * dimY * dimZ, 0.0);
  }

  double &operator()(int x, int y, int z)
  {
    return points.at(z * (dimX * dimY) + y * dimX + x);
  }

  double operator()(int x, int y, int z) const
  {
    return points.at(z * (dimX * dimY) + y * dimX + x);
  }

  void fill(double val)
  {
    for (int i = 0; i < dimX * dimY * dimZ; i++)
      points[i] = val;
  }

  int dimX, dimY, dimZ;
  std::vector<double> points;
};

static inline double absd(double v)
{
  return (v > 0.0) ? v : (-v);
}

double grid_error(const PointGrid &g1, const PointGrid &g2)
{
  double error = 0.0;
  for (int i = 0; i < g1.dimX * g1.dimY * g1.dimZ; i++)
  {
    double v = (g1.points[i] - g2.points[i]);
    v = (v < 0.0)? (-v) : v; 
    error = (v > error) ? v : error;
  }

  return error;
}

void grid_difference(PointGrid &out, const PointGrid &g1, const PointGrid &g2)
{
  for (int i = 0; i < g1.dimX * g1.dimY * g1.dimZ; i++)
  {
    out.points[i] = absd(g1.points[i] - g2.points[i]);
  }
}

//yaml
void save_grid(const SolveParams &p, const PointGrid &g, int time_step, const char *grid_name)
{
  std::stringstream file_name;
  file_name << grid_name << "_" << time_step << ".yaml";
  
  std::string out_name = file_name.str();
  std::ofstream out(out_name.c_str());
  
  out << "name : " << grid_name << std::endl;
  out << "L : " << p.Lx << std::endl;
  out << "T : " << p.T << std::endl;
  out << "time_steps : " << p.time_steps << std::endl;
  out << "time_step : " << time_step << std::endl; 
  out << "dim : " << g.dimX << std::endl;
  out << "points:" << std::endl;

  for (int i = 0; i < g.points.size(); i++)
    out << "- " << g.points[i] << std::endl;
  out.close();
}

// variant 6, x = П y = 1Р z = П
// x: u(0, y, z, t) = u(Lx, y, z, t); dudx(0, y, z, t) = dudx(Lx, y, z, t)
// y: u(x, 0, z, t) = 0; u(x, Ly, z, t) = 0
// z: u(x, y, 0, t) = u(x, y, Lz, t); dudz(x, y, 0, t) = dudz(x, y, Lz, t)

typedef double (*TargetFunction)(const SolveParams &p, double x, double y, double z, double t);

double target_function(const SolveParams &p, double x, double y, double z, double t)
{
  double v1 = sin(2.0 * M_PI/p.Lx * x);
  double v2 = sin(M_PI/p.Ly * y + M_PI);
  double v3 = sin(2.0 * M_PI/p.Lz * z + 2.0 * M_PI);

  const double at = (M_PI/3.0) * sqrt(4.0/(p.Lx * p.Lx) + 1.0/(p.Ly * p.Ly) + 4.0/(p.Lz * p.Lz));
  double v4 = cos(at * t + M_PI);

  return  v1 * v2 * v3 * v4;
}

double laplacian(const SolveParams &p, const PointGrid &g, int x, int y, int z)
{
  double res = 0.0;
  double h2 = p.h * p.h;
  double uc = g(x, y, z);

  int x1 = (x == 0)? (g.dimX - 2) : (x - 1);
  int x2 = (x == g.dimX - 1)? 1 : x + 1;

  int z1 = (z == 0)? (g.dimZ - 2) : (z - 1);
  int z2 = (z == g.dimZ - 1)? 1 : z + 1;

  res += g(x1, y, z) - 2.0 * uc + g(x2, y, z);
  res += g(x, y - 1, z) - 2.0 * uc + g(x, y + 1, z);
  res += g(x, y, z1) - 2.0 * uc + g(x, y, z2);

  return res/h2; 
}

void fill_grid(const SolveParams &p, PointGrid &g, double t)
{
  #pragma omp parallel for
  for (int zs = 0; zs < p.dim_steps; zs++)
  {
    #pragma omp parallel for
    for (int ys = 0; ys < p.dim_steps; ys++)
    {
      #pragma omp parallel for
      for (int xs = 0; xs < p.dim_steps; xs++)
      {
        double x = xs * p.h;
        double y = ys * p.h;
        double z = zs * p.h;
        g(xs, ys, zs) = target_function(p, x, y, z, t);
      }
    }
  }
}

RunStats run_pass(double dim_size, int dim_steps, double max_t, int time_steps, int max_threads, bool check_error, bool save_file)
{
  SolveParams params;
  params.Lx = params.Ly = params.Lz = dim_size;
  params.a2 = 1.0/9.0; // variant 6
  params.dim_steps = dim_steps;
  params.num_threads = max_threads;
  params.T = max_t;
  params.time_steps = time_steps;

  params.init();

  omp_set_num_threads(params.num_threads);

  RunStats results;
  results.src = params;
  results.error = 0.0;
  results.init_time_ms = 0.0;
  results.iter_time_ms = 0.0;

  TimeProfiler init_counter(results.init_time_ms);


  PointGrid referenceGrid(params.dim_steps, params.dim_steps, params.dim_steps);
  
  std::vector<PointGrid> grids(3);
  //3 steps
  for (int i = 0; i < 3; i++)
    grids[i].init(params.dim_steps, params.dim_steps, params.dim_steps);

  //init n-2
  fill_grid(params, grids[0], 0.0); // init equation state. u = phi(..)

  if (save_file)
  {
    save_grid(params, grids[0], 0, "out/reference");
  }
  //init N - 1
  #pragma omp parallel for
  for (int zs = 0; zs < params.dim_steps; zs++)
  {
    #pragma omp parallel for
    for (int ys = 1; ys < params.dim_steps - 1; ys++)
    {
      #pragma omp parallel for
      for (int xs = 0; xs < params.dim_steps; xs++)
      {      
        grids[1](xs, ys, zs) = grids[0](xs, ys, zs) 
          + 0.5 * params.a2 * params.tau * params.tau * laplacian(params, grids[0], xs, ys, zs); 
      }
    }
  }
  
  init_counter.stop();

  if (check_error)
  {
    fill_grid(params, referenceGrid, params.tau);
    double err = grid_error(referenceGrid, grids[1]);
    results.error = (err > results.error)? err : results.error; 
  }
  
  if (save_file)
  {
    save_grid(params, grids[1], 1, "out/computed");
    if (check_error)
      save_grid(params, referenceGrid, 1, "out/reference");
  }

  // N <- N - 1, N - 2

  for (int ts = 2; ts < params.time_steps; ts++)
  {
    TimeProfiler iter_counter(results.iter_time_ms);

    PointGrid &g = grids[ts % 3];
    PointGrid &g1 = grids[(ts - 1) % 3];
    PointGrid &g2 = grids[(ts - 2) % 3];
    
    std::cout << "Running iter = " << ts << std::endl;

    #pragma omp parallel for 
    for (int zs = 0; zs < params.dim_steps; zs++)
    {
      #pragma omp parallel for
      for (int ys = 1; ys < params.dim_steps - 1; ys++)
      {
        #pragma omp parallel for
        for (int xs = 0; xs < params.dim_steps; xs++)
        {
          double un = g1(xs, ys, zs);
          double un_1 = g2(xs, ys, zs);
          double l_un = laplacian(params, g1, xs, ys, zs);
          g(xs, ys, zs) = params.tau * params.tau * (params.a2 * l_un) + 2.0 * un - un_1;
        }
      }
    }

    iter_counter.stop();

    if (check_error)
    {
      fill_grid(params, referenceGrid, params.tau * ts);
      double err = grid_error(referenceGrid, g);
      results.error = (err > results.error)? err : results.error; 
    }
    
    if (save_file && ((ts - 2) % 20 == 0 || ts == params.time_steps - 1))
    {
      save_grid(params, g, ts, "out/computed");
      if (check_error)
        save_grid(params, referenceGrid, ts, "out/reference");
    }
  }

  return results;
}

struct BenchParams
{
  BenchParams(double L_, int dim_, int t_steps_, int threads_)
    : L(L_), dim(dim_), t_steps(t_steps_), threads(threads_)
  {
    double h = L/(dim - 1);
    double t_step = h/10.0; // error not grows too fast
    t_max = t_step * (t_steps - 1);
  }

  double L;
  int dim;
  double t_max;
  int t_steps;
  int threads; 
};

void dump_result(const RunStats &res)
{
  std::cout << "L = " << res.src.Lx << "\n";
  std::cout << "dim = " << res.src.dim_steps << "\n";
  std::cout << "T = " << res.src.T << "\n";
  std::cout << "time steps = " << res.src.time_steps << "\n";
  std::cout << "threads  = " << res.src.num_threads << "\n";
  std::cout << "Init time = " << res.init_time_ms << "ms\n";
  std::cout << "Iter time = " << res.iter_time_ms << "ms\n";
  std::cout << "Max error = " << res.error << "\n";
}

void benchmark()
{
  std::vector<RunStats> benchmark_results;
  std::vector<BenchParams> init_params;

  BenchParams b1(1.0, 128, 50, 1);
  BenchParams b2(1.0, 256, 50, 1);
  BenchParams b3(M_PI, 128, 50, 1);
  BenchParams b4(M_PI, 256, 50, 1);

  init_params.push_back(b1);
  init_params.push_back(b2);
  init_params.push_back(b3);
  init_params.push_back(b4);

  const int max_threads = omp_get_max_threads();
  std::cout << "Platform supports max " << max_threads << " threads\n";
  std::vector<int> thread_params;

  int threads_count = 1;
  
  do 
  {
    threads_count *= 2;
    if (threads_count > max_threads)
      threads_count = max_threads;
    thread_params.push_back(threads_count);
  } while (threads_count < max_threads);
  
  for (int b_id = 0; b_id < init_params.size(); b_id++)
  {
    std::cout << "Running benchmark " << (b_id+1) << "/" << init_params.size() << "\n";

    BenchParams &bench = init_params[b_id];
    //baseline
    benchmark_results.push_back(run_pass(bench.L, bench.dim, bench.t_max, bench.t_steps, 1, true, false));

    // estimate acceleration
    for (int i = 0; i < thread_params.size(); i++)
    {
      benchmark_results.push_back(
        run_pass(bench.L, bench.dim, bench.t_max, bench.t_steps, thread_params[i], true, false));
    }
  }

  // output
  for (int run_id = 0; run_id < benchmark_results.size(); run_id++)
  {
    const RunStats &res = benchmark_results[run_id]; 
    std::cout << "Pass " << run_id << "\n";
    dump_result(res);
  }
}

int main(int argc, char **argv)
{
#ifdef RUN_BENCHMARK
  benchmark();
#else
  const int max_threads = omp_get_max_threads();
  std::cout << "Platform supports max " << max_threads << " threads\n";

  BenchParams p(1.0, 128, 1000, 8);
  auto res = run_pass(p.L, p.dim, p.t_max, p.t_steps, 8, true, false);
  dump_result(res);
#endif
  return 0;
}