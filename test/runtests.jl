using LinearAlgebra, Random, Base.Threads, Test
using MPI, Distributed, MPIClusterManagers
using ReorderedStaticCondensation

MPI.Init(;threadlevel=MPI.THREAD_SERIALIZED)
const cmm = MPI.COMM_WORLD
const rnk = MPI.Comm_rank(cmm)
const sze = MPI.Comm_size(cmm)
const nts = Threads.nthreads()

using Random
Random.seed!(0)

# [A1 0  0  :    C1  ] x1     b1    }rank 0
#   0 A2 0  :    C2  ] x2     b2    }rank 0
#   0 0  \  :    :   ]  :   =  :
#   ..   .. An-1 Cn-1] xn-1   bn-1  }rank commsize-1
#  B1 B2 .. Bn-1 An  ] xn     bn    }rank commsize-1

function setupmatrix(;nglobalblocks, nlocalblocks, blocksize=4, couplingsize=3)
  N = nglobalblocks
  bs = blocksize
  cs = couplingsize

  Ai = Vector{Matrix{Float64}}()
  Bi = Vector{Matrix{Float64}}()
  Ci = Vector{Matrix{Float64}}()
  bi = Vector{Vector{Float64}}()
  D = Matrix{Float64}(undef, 0, 0)

  x = Float64[]
  schurrank = sze-1
  x = if rnk == schurrank
    Ai = [rand(bs, bs) for i in 1:N-1]
    Ci = [rand(bs, cs) for i in 1:N-1]
    Bi = [rand(cs, bs) for i in 1:N-1]
    D = rand(cs, cs)
    bi = [[rand(bs) for i in 1:N-1]..., rand(cs)]

    lhs = zeros(bs * (N-1) + cs, bs * (N-1) + cs)
    for i in eachindex(Ai)
      lhs[bs*(i-1)+1:bs*i, bs*(i-1)+1:bs*i] .= Ai[i]
      lhs[bs*(i-1)+1:bs*i, bs*(N-1)+1:end] .= Ci[i]
      lhs[bs*(N-1)+1:end, bs*(i-1)+1:bs*i] .= Bi[i]
    end
    lhs[bs*(N-1)+1:end, bs*(N-1)+1:end] .= D
    rhs = vcat(bi...)
    x = lhs \ rhs
  else
    nothing
  end
  x = MPI.bcast(x, cmm; root=schurrank)
  L = nlocalblocks
  if rnk == schurrank
    for r in 0:sze-1
      r == schurrank && continue
      for i in r*L+1:(r + 1)*L
        MPI.send(Ai[i], cmm; dest=r, tag=0N + i * sze + r)
        MPI.send(Bi[i], cmm; dest=r, tag=1N + i * sze + r)
        MPI.send(Ci[i], cmm; dest=r, tag=2N + i * sze + r)
        MPI.send(bi[i], cmm; dest=r, tag=3N + i * sze + r)
      end
    end
  else
    r = rnk
    for i in r*L+1:(r + 1)*L
      push!(Ai, MPI.recv(cmm; source=schurrank, tag=0N + i * sze + r))
      push!(Bi, MPI.recv(cmm; source=schurrank, tag=1N + i * sze + r))
      push!(Ci, MPI.recv(cmm; source=schurrank, tag=2N + i * sze + r))
      push!(bi, MPI.recv(cmm; source=schurrank, tag=3N + i * sze + r))
    end
  end
  dels = Int[]
  if rnk == schurrank
    for r in 0:sze-1, i in r*L+1:(r + 1)*L
      r == schurrank && continue
      push!(dels, i)
    end
  end
  deleteat!(Ai, dels)
  deleteat!(Bi, dels)
  deleteat!(Ci, dels)
  deleteat!(bi, dels)

  return Ai, Bi, Ci, D, bi, x
end

function run(nlocalblocks=2; nglobalblocks=nlocalblocks*sze, blocksize=4, couplingsize=3)
  Ai, Bi, Ci, D, bi, expected = setupmatrix(;
    nglobalblocks=nglobalblocks, nlocalblocks=nlocalblocks,
    blocksize=blocksize, couplingsize=couplingsize)
  context = ReorderedStaticCondensation.MPIContext(cmm, rnk, sze, ReorderedStaticCondensation.DistributedMemoryMPI())
  M = ReorderedStaticCondensation.RSCMatrix(Ai, Bi, Ci, D; globalcontext=context)
  luM = lu!(M)

  b = vcat(bi...)
  x = deepcopy(b)
  ldiv!(x, luM, b)

  xcounts = zeros(Int32, sze)
  MPI.Allgatherv!(Int32[length(x)], xcounts, ones(Int32, sze), cmm)
  result = zeros(eltype(expected), size(expected)...)
  MPI.Gatherv!(x, result, xcounts, 0, cmm)
  rnk == 0 && @test result ≈ expected
end

@testset "ReorderedStaticCondensation.jl" begin
  run(2; blocksize=2, couplingsize=1)
  run(2; blocksize=4, couplingsize=3)
  run(3; blocksize=4, couplingsize=3)
  run(4; blocksize=16, couplingsize=5)
end
