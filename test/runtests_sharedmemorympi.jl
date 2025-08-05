using LinearAlgebra, Random, Base.Threads, Test
using MPI, Distributed, MPIClusterManagers
using ReorderedStaticCondensation

MPI.Init(;threadlevel=MPI.THREAD_SERIALIZED)
const globalcmm = MPI.COMM_WORLD
const globalrnk = MPI.Comm_rank(globalcmm)
const globalsze = MPI.Comm_size(globalcmm)
const nts = Threads.nthreads()

const targetlocalcommsize = max(1, globalsze ÷ 2)
const key = mod(globalrnk, targetlocalcommsize)
const colour = globalrnk ÷ targetlocalcommsize

const localcmm = MPI.Comm_split(globalcmm, colour, key)
const localrnk = MPI.Comm_rank(localcmm)
const localsze = MPI.Comm_size(localcmm)

using Random
Random.seed!(0)
#fetch.([Threads.@spawn ()->Random.seed!(0) for i in 1:nthreads()]);

# [A1 0  0  :    C1  ] x1     b1    }rank 0
#   0 A2 0  :    C2  ] x2     b2    }rank 0
#   0 0  \  :    :   ]  :   =  :
#   ..   .. An-1 Cn-1] xn-1   bn-1  }rank commsize-1
#  B1 B2 .. Bn-1 An  ] xn     bn    }rank commsize-1

function allocmemory(shared_size, local_rank, local_comm, T=Float64; sharedmem=false)
  if sharedmem
    win, ptr = MPI.Win_allocate_shared(T, prod(shared_size), local_comm)
    _, _, base_ptr = MPI.Win_shared_query(win, 0)
    base_ptr = Ptr{T}(base_ptr)
    array = unsafe_wrap(Matrix{T}, base_ptr, shared_size)
    fill!(array, zero(T))
    return array, win
  else
    return zeros(T, shared_size...), nothing
  end
end

function setupmatrix(;nlocalblocks, blocksize=4, couplingsize=3, sharedmem=false)

  schurrank = globalsze-1

  bs = blocksize
  cs = couplingsize

  D = Matrix{Float64}(undef, 0, 0)

  indices = UnitRange[]
  indicesowners = vcat((r .* ones(Int, nlocalblocks) for r in 0:(globalsze-1))...)
  N = nlocalblocks * globalsze
  a = 1
  for i in 1:(N - 1)
    b = a + blocksize - 1
    push!(indices, a:b)
    a += blocksize
  end
  push!(indices, a:a+couplingsize-1)
  
  nelems = [length(i) for i in indices]
  n = sum(nelems)
  A = zeros(n, n)
  b = zeros(n, 1)
  x = if globalrnk == schurrank
    for i in eachindex(indices)[1:end-1]
      A[indices[i], indices[i]] .= rand(bs, bs)
      A[indices[end], indices[i]] .= rand(cs, bs) # B
      A[indices[i], indices[end]] .= rand(bs, cs) # C
    end
    D = rand(cs, cs)
    A[indices[end], indices[end]] .= D
    b .= rand(size(b)...)
    x = A \ b
  else
    Float64[]
  end
  A = MPI.bcast(A, globalcmm; root=schurrank)
  x = MPI.bcast(x, globalcmm; root=schurrank)
  b = MPI.bcast(b, globalcmm; root=schurrank)
  L = nlocalblocks

  wins = sharedmem ? MPI.Win[] : Any[]
  Ai = Matrix{Float64}[]
  Bi = Matrix{Float64}[]
  Ci = Matrix{Float64}[]
  for ii in 1:L
    i = globalrnk * L + ii
    Aii, winA = allocmemory((nelems[i], nelems[i]), localrnk, localcmm; sharedmem=sharedmem)
    Bii, winB = allocmemory((nelems[end], nelems[i]), localrnk, localcmm; sharedmem=sharedmem)
    Cii, winC = allocmemory((nelems[i], nelems[end]), localrnk, localcmm; sharedmem=sharedmem)
    push!(wins, winA)
    push!(wins, winB)
    push!(wins, winC)
    ((ii == L) && (globalrnk == schurrank)) && continue
    push!(Ai, Aii)
    Ai[ii] .= A[indices[i], indices[i]]
    push!(Bi, Bii)
    Bi[ii] .= A[indices[end], indices[i]]
    push!(Ci, Cii)
    Ci[ii] .= A[indices[i], indices[end]]
  end
  globalindices = indices[indicesowners .== globalrnk]
  blocal, win = allocmemory((sum(length.(globalindices)), size(b, 2)), localrnk, localcmm;
    sharedmem=sharedmem)
  push!(wins, win)
  blocal .= b[first(globalindices[1]):last(globalindices[end]), :]

  return Ai, Bi, Ci, D, blocal, wins, x
end

function run(nlocalblocks=2; sharedmem=false, blocksize=4, couplingsize=3)
  Ai, Bi, Ci, D, b, wins, expected = setupmatrix(; sharedmem=sharedmem,
    nlocalblocks=nlocalblocks, blocksize=blocksize, couplingsize=couplingsize,)
  globalcontext = ReorderedStaticCondensation.MPIContext(
    ReorderedStaticCondensation.DistributedMemoryMPI(),
    globalcmm, globalrnk, globalsze)
  localcontext = if sharedmem
    ReorderedStaticCondensation.MPIContext(
      ReorderedStaticCondensation.SharedMemoryMPI(wins),
      localcmm, localrnk, localsze)
  else
    ReorderedStaticCondensation.ThreadedContext()
  end
  sleep(globalrnk)
  @show Ai
  @show Bi
  @show Ci
  @show D
  MPI.Barrier(globalcmm)
  M = ReorderedStaticCondensation.RSCMatrix(Ai, Bi, Ci, D;
    globalcontext=globalcontext, localcontext=localcontext)
  luM = lu!(M)
  sleep(globalrnk)
  @show globalrnk, M.schurrank
  @show luM.A.localAii
  @show luM.A.localBi
  @show luM.A.localCi
  @show luM.A.D
  MPI.Barrier(globalcmm)
  error("adsgagaw")
  sleep(globalrnk)

  x = similar(b)
  fill!(x, 0) # doesn't need to be filled with b values, but could if deepcopy is easier
  ldiv!(x, luM, b)

  xcounts = zeros(Int32, globalsze)
  MPI.Allgatherv!(Int32[length(x)], xcounts, ones(Int32, globalsze), globalcmm)
  result = zeros(eltype(expected), size(expected)...)
  MPI.Gatherv!(x, result, xcounts, 0, globalcmm)
  globalrnk == 0 && @test result ≈ expected
end
# This code is allocating memory for the shared memory MPI as though
# there is a single global rank per shared mem communicator i.e.
# several shared mem ranks share a distributed mem rank.
# The logic then acts as though there are unique distributed
# communicator ranks across all processes, which isn't the case.
# If there are unique distributed communicator ranks for every process,
# then there is no gain of having shared memory MPI.
@testset "ReorderedStaticCondensation.jl" begin
  # Threaded tests
  #for b in (1, 2, 4, 6), c in (1, 2, 5)
  #  run(2; blocksize=b, couplingsize=c)
  #  run(3; blocksize=b, couplingsize=c)
  #  run(4; blocksize=b, couplingsize=c)
  #end

  # MPI Shared memory tests
  for b in (1, 2, 4, 6), c in (1, 2, 5)
    run(2; blocksize=b, couplingsize=c, sharedmem=true)
    run(3; blocksize=b, couplingsize=c, sharedmem=true)
    run(4; blocksize=b, couplingsize=c, sharedmem=true)
  end
end
