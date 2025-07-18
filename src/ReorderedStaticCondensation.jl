module ReorderedStaticCondensation

using Base.Threads
using LinearAlgebra
using MPI

abstract type AbstractMPI end
struct DistributedMemoryMPI <: AbstractMPI end
struct SharedMemoryMPI <: AbstractMPI
  wins::Vector{MPI.Win}
end
free(w::SharedMemoryMPI) = for w in w.wins; MPI.free(w); end

abstract type AbstractContext end

struct ThreadedContext{C} <: AbstractContext
  comm::C
end
ThreadedContext() = ThreadedContext(-1)

MPI.Comm_rank(con::ThreadedContext) = 0

struct MPIContext{C, T<:AbstractMPI} <: AbstractContext
  comm::C
  rank::Int
  size::Int
  mpitype::T
  function MPIContext(comm::C, rank::Int, size::Int, mpitype::T) where {C, T<:AbstractMPI}
    @assert 0 <= rank < size
    return new{C, T}(comm, rank, size, mpitype)
  end
end

MPI.Comm_rank(con::MPIContext) = con.rank

Base.sum(a::Int, con::ThreadedContext) = a
Base.sum(a::Int, con::MPIContext) = MPI.AllReduce(a, +, con.comm)

free(con::AbstractContext) = nothing
free(con::MPIContext{SharedMemoryMPI}) = free(con.mpitype)

lutype(A::Matrix{T}) where T = LU{T, Matrix{T}, Vector{Int64}}

struct RSCMatrix{T<:Number, C1<:AbstractContext, C2<:Union{Nothing, AbstractContext}}
  localAii::Vector{Matrix{T}}
  localBi::Vector{Matrix{T}}
  localCi::Vector{Matrix{T}}
  D::Matrix{T}
  nblocksglobal::Int
  nblockslocal::Int
  blockranks::Vector{Int32} # 1:nglobalblocks containing the rank of the owner of each block
  globalcontext::C1
  localcontext::C2
  lulocalAii::Dict{Int, LU{T, Matrix{T}, Vector{Int64}}}
  schurrank::Int32
  function RSCMatrix(localAii::Vector{<:AbstractMatrix{T1}},
                     localBi::Vector{<:AbstractMatrix{T1}},
                     localCi::Vector{<:AbstractMatrix{T1}},
                     D=Matrix{T1}(undef, 0, 0);
                     globalcontext=ThreadedContext(),
                     localcontext=nothing) where T1

    globalrank = MPI.Comm_rank(globalcontext)

    # owner of the lower row of Bi, and crucially the lower right block
    schurowner = !iszero(length(D))
    schurranks = [schurowner * globalrank]

    schurrank = MPI.Allreduce!(schurranks, +, globalcontext.comm)[1]

    commsize = MPI.Comm_size(globalcontext.comm)
    nblockslocal = Int(length(localAii) + schurowner)
    nblocksglobal = MPI.Allreduce(nblockslocal, +, globalcontext.comm)

    nlocalblocksall = zeros(Int32, commsize)
    nlocalblocksall[globalrank + 1] = nblockslocal
    MPI.Allgather!(nlocalblocksall, Int32(1), globalcontext.comm)
    blockranks = vcat((ones(Int32, nlocalblocksall[i]) .* (i - 1) for i in eachindex(nlocalblocksall))...)

    for Aii in localAii
      @assert size(Aii, 1) == size(Aii, 2)
    end
    @assert length(blockranks) == nblocksglobal
    @assert nblockslocal - 1 <= length(localBi) <= nblockslocal "0 <= $(length(localBi)) <= $(nblockslocal)"
    @assert length(localBi) == length(localCi)

    T = eltype(eltype(localAii))
    LUT = LU{T, Matrix{T}, Vector{Int64}}

    return new{T, typeof(globalcontext), typeof(localcontext)}(
      localAii, localBi, localCi, D,
      nblocksglobal, nblockslocal, blockranks, globalcontext, localcontext,
      Dict{Int, LUT}(), schurrank)
  end
end

struct RSCMatrixLU{T, C1, C2}
  A::RSCMatrix{T, C1, C2}
  localAii_indices::Vector{UnitRange{Int}}
  function RSCMatrixLU(A::RSCMatrix{T, C1, C2}
      ) where {T, C1<:AbstractContext, C2<:Union{Nothing, AbstractContext}}
    Aiisizes = [size(Aii, 1) for Aii in A.localAii]
    @views localAii_indices = [sum(Aiisizes[1:i-1])+1:sum(Aiisizes[1:i])
                               for i in eachindex(Aiisizes)]
    return new{T,C1,C2}(A, localAii_indices)
  end
end

function LinearAlgebra.lu!(A::RSCMatrix)

  globalrank = MPI.Comm_rank(A.globalcontext)
#  # 1. LU decompose A
#  luAi = [lu!(localAii) for localAii in A.localAii] # each done locally on communicator
#  # 2. Calculate partial solution of A \ C
#  C̃i = [luAi[i] \ localCi[i] for i in 1:N] 
#  # 3. Calculate Schur complement
#  S = D - sum(Bi[i] * C̃i[i] for i in 1:N)
  for (i, Aii) in enumerate(A.localAii)
    A.lulocalAii[i] = lu!(Aii) # 1.
    ldiv!(A.localCi[i], A.lulocalAii[i], A.localCi[i]) # 2.
  end

  ∑BiC̃i = sum(A.localBi[i] * A.localCi[i] for i in eachindex(A.localCi)) # Part of 3

  ∑BiC̃i = MPI.Reduce(∑BiC̃i, +, A.globalcontext.comm; root=A.schurrank) # Part of 3.
  if globalrank == A.schurrank
    A.D .-= ∑BiC̃i # 3
  end

  return RSCMatrixLU(A)
end
function _ldivbloop!(b, A::RSCMatrixLU{T, C1, C2}) where {T, C1, C2}
  @threads for (i, is) in collect(enumerate(A.localAii_indices))
    bi = view(b, is, :)
    ldiv!(bi, A.A.lulocalAii[i], bi)
  end
end
function _ldivbloop!(b, A::RSCMatrixLU{T, C1, MPIContext{SharedMemoryMPI}}) where {T, C1}
  localsze = MPI.Comm_size(A.A.localcontext.comm)
  localrnk = MPI.Comm_rank(A.A.localcontext.comm)
  for (i, is) in enumerate(A.localAii_indices)
    rem(i, localsze) == localrnk || continue
    bi = view(b, is, :)
    ldiv!(bi, A.A.lulocalAii[i], bi)
  end
end


function isschurrank(A::RSCMatrixLU)
  return A.A.schurrank == MPI.Comm_rank(A.A.globalcontext.comm)
end
function isschurrank(A::RSCMatrixLU{T, C1, MPIContext{SharedMemoryMPI}}) where {T, C1}
  return (A.A.schurrank == MPI.Comm_rank(A.A.globalcontext.comm)) &&
    (MPI.Comm_rank(A.A.localcontext.comm) == 0)
end

function _solverupperx!(x, b, A, y)
  N = length(A.A.localAii)
  @threads for (i, is) in collect(enumerate(A.localAii_indices[1:N]))
    @views x[is, :] .= b[is, :] - A.A.localCi[i] * y
  end
end

function _solverupperx!(x, b, A::RSCMatrixLU{T, C1, MPIContext{SharedMemoryMPI}}, y
    ) where {T, C1}
  N = length(A.A.localAii)
  localsze = MPI.Comm_size(A.A.localcontext.comm)
  localrnk = MPI.Comm_rank(A.A.localcontext.comm)
  for (i, is) in enumerate(A.localAii_indices[1:N])
    rem(i, localsze) == localrnk || continue
    @views x[is, :] .= b[is, :] - A.A.localCi[i] * y
  end
end

function LinearAlgebra.ldiv!(x, A::RSCMatrixLU{T}, b::AbstractArray) where T
  @assert size(x, 1) == size(b, 1)
  @assert size(x, 2) == size(b, 2)
  x .= b
  # 4. Calculate partial solution of A \ b
  _ldivbloop!(b, A)
   # 5. Calculate right hand side for Schur complement solve
  # c̃ = c - sum(A.localBi[i] * b̃i[i] for i in 1:N)
  N = length(A.A.localAii)

  # this is the bit that could be parallelised by threads / a sub-communicator
  ∑Bibi = sum(A.A.localBi[i] * b[A.localAii_indices[i], :] for i in 1:N)
  # now reduce to the Schur rank
  ΣBibi = MPI.Reduce(∑Bibi, +, A.A.globalcontext.comm; root=A.A.schurrank)
  y = if isschurrank(A)
    @views c = b[A.localAii_indices[end][end] + 1:end, :] .- ΣBibi
    # 6. Solve Schur complement to get lower part of x vector
    A.A.D \ c
  else
    zeros(T, zeros(Int, size(x))...)
  end
  y = MPI.bcast(y, A.A.globalcontext.comm; root=A.A.schurrank)

  # 7. Solve for upper parts of x vector
  _solverupperx!(x, b, A, y)
  if isschurrank(A)
    x[A.localAii_indices[end][end] + 1:end, :] .= y
  end
  return x
end
end # module ReorderedStaticCondensation
