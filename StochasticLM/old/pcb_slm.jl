using SparseArrays, ForwardDiff, LinearOperators, Test, Random

function makeop(M::AbstractMatrix{T}, N::AbstractMatrix{T}, U::AbstractMatrix{T}, Vt::AbstractMatrix{T},
                Sw::AbstractMatrix{T}, Sh::AbstractMatrix{T}, Gw::AbstractMatrix{T}, Gh::AbstractMatrix{T}) where T
    m′, k′ = size(U); k″, n′ = size(Vt)
    r′, p′ = size(M); p″, r″ = size(N)
    @assert (k′ == k″) && (k′ == r′) && (r′ == r″) && (p′ == p″)
    return LinearOperator{T}(r′*r″+2*(m′+n′)*p′, (r′ + r″) * p′, false, false,
            function(res, v, α, β)    # Jacobian-vector product
                dM = reshape(v[1:r′*p′], r′, p′)
                dN = reshape(v[r′*p′+1:end], p″, r″)
                Ys = reshape(view(res,1:r′*r″), r′, r″)
                mul!(Ys, dM, N, α, β)
                mul!(Ys, M, dN, α, true)
                Ynw = reshape(view(res,r′*r″+1:r′*r″+m′*p′), m′, p′)
                Ynh = reshape(view(res,r′*r″+m′*p′+1:r′*r″+m′*p′+p″*n′), p″, n′)
                Ysw = reshape(view(res,r′*r″+m′*p′+p″*n′+1:r′*r″+m′*p′+p″*n′+m′*p′), m′, p′)
                Ysh = reshape(@view(res[r′*r″+m′*p′+p″*n′+m′*p′+1:end]), p″, n′)
                mul!(Ynw, U, dM, α, β); copyto!(Ysw,Ynw); Ynw .*= Sw; Ysw .*= Gw # dYnw=Sw⨀(W*dM), dYsw=Gw⨀(W*dM)
                mul!(Ynh, dN, Vt, α, β); copyto!(Ysh,Ynh); Ynh .*= Sh; Ysh .*= Gh  # dYnh=Sh⨀(dN*Vt), dYsh=Gh⨀(dN*Vt)
                return res
            end,
            function(res, v, α, β)  # Jacobian-transpose-vector product
                dYs = reshape(v[1:r′*r″], r′, r″)
                dYnw = reshape(v[r′*r″+1:r′*r″+m′*p′], m′, p′)
                dYnh = reshape(v[r′*r″+m′*p′+1:r′*r″+m′*p′+p″*n′], p″, n′)
                dYsw = reshape(v[r′*r″+m′*p′+p″*n′+1:r′*r″+m′*p′+p″*n′+m′*p′], m′, p′)
                dYsh = reshape(v[r′*r″+m′*p′+p″*n′+m′*p′+1:end], p″, n′)
                dM = reshape(view(res,1:r′*p′), r′, p′)
                dN = reshape(@view(res[r′*p′+1:end]), p″, r″)
                mul!(dM, dYs, N', α, β); dYnw .*= Sw; mul!(dM, U', dYnw, α, true)
                dYsw .*= Gw; mul!(dM, U', dYsw, α, true)  # dYs*N' + U'*(Sw⨀dYnw) + U'*(Gw⨀dYsw)
                mul!(dN, M', dYs, α, β); dYnh .*= Sh; mul!(dN, dYnh, Vt', α, true)
                dYsh .*= Gh; mul!(dN, dYsh, Vt', α, true)  # M'*dYs + (Sh⨀dYnh)*Vt' + (Gh⨀dYsh)*Vt'
                return res
            end,
            nothing)
end

g(X::AbstractMatrix, σ) = (X.^2 .+ σ^2).^(1/4)
dg(X::AbstractMatrix, σ) = (0.5*(X.^2 .+ σ^2).^(-3/4)).*X

function make_fcache_op!(M::AbstractMatrix{T}, N::AbstractMatrix{T}, U::AbstractMatrix{T}, Vt::AbstractMatrix{T},
                        D::AbstractMatrix{T}, Sw::AbstractMatrix{T}, Sh::AbstractMatrix{T}, 
                        Gw::AbstractMatrix{T}, Gh::AbstractMatrix{T}, rtβw, rtβh, rtαw, rtαh, σw, σh) where T
    p = size(M,2); m, k = size(U); n = size(Vt, 2)
    x = [vec(D); zeros(m*p); zeros(n*p); zeros(m*p); zeros(n*p)]
    xpred = copy(x)
    Dpred = reshape(view(xpred,1:k^2),k,k)
    Ynwpred = reshape(view(xpred,k^2+1:k^2+m*p),m,p)
    Ynhpred = reshape(view(xpred,k^2+m*p+1:k^2+(m+n)*p),p,n)
    Yswpred = reshape(view(xpred,k^2+(m+n)*p+1:k^2+(2m+n)*p),m,p)
    Yshpred = reshape(@view(xpred[k^2+(2m+n)*p+1:end]),p,n)
    Mscratch, Nscratch, xpredscratch = copy(M), copy(N), copy(x)
    Dpredscratch = reshape(view(xpredscratch,1:k^2),k,k)
    Ynwpredscrach = reshape(view(xpredscratch,k^2+1:k^2+m*p),m,p)
    Ynhpredscrach = reshape(view(xpredscratch,k^2+m*p+1:k^2+(m+n)*p),p,n)
    Yswpredscrach = reshape(view(xpredscratch,k^2+(m+n)*p+1:k^2+(2m+n)*p),m,p)
    Yshpredscrach = reshape(@view(xpredscratch[k^2+(2m+n)*p+1:end]),p,n)
    return function(rj, θ, idx)
        if rj === nothing
            # Don't modify M, N, Dpred
            copyto!(Mscratch, view(θ, 1:k*p))
            copyto!(Nscratch, @view(θ[k*p+1:end]))
            mul!(Dpredscratch, Mscratch, Nscratch)
            mul!(Ynwpredscrach, U, Mscratch); copyto!(Yswpredscrach, Ynwpredscrach)
            copyto!(Sw, (Ynwpredscrach.<0)*rtβw); Ynwpredscrach .*= Sw
            copyto!(Yswpredscrach, g(Yswpredscrach, σw)*rtαw)
            mul!(Ynhpredscrach, Nscratch, Vt); copyto!(Yshpredscrach, Ynhpredscrach)
            copyto!(Sh, (Ynhpredscrach.<0)*rtβh); Ynhpredscrach .*= Sh 
            copyto!(Yshpredscrach, g(Yshpredscrach, σh)*rtαh)
            return sum((x[i] - xpredscratch[i])^2 for i in idx) / 2
        end
        copyto!(M, view(θ, 1:k*p))
        copyto!(N, @view(θ[k*p+1:end]))
        mul!(Dpred, M, N)
        mul!(Ynwpred, U, M); copyto!(Yswpred, Ynwpred)
        copyto!(Sw,(Ynwpred.<0)*rtβw); Ynwpred .*= Sw
        copyto!(Gw, dg(Yswpred, σw)*rtαw); copyto!(Yswpred, g(Yswpred, σw)*rtαw)
        mul!(Ynhpred, N, Vt); copyto!(Yshpred, Ynhpred)
        copyto!(Sh, (Ynhpred.<0)*rtβh); Ynhpred .*= Sh 
        copyto!(Gh, dg(Yshpred, σh)*rtαh); copyto!(Yshpred, g(Yshpred, σh)*rtαh)
        @inbounds r = x[idx] - xpred[idx]
        rj.r[idx] = r
        #Cache Hd
        # for j = 1:p
        #     s = sum(abs2, @view(N[j, :]))
        #     for i = 1:k
        #         rj.Hd[i + (j - 1) * k] = s
        #     end
        # end
        # offset = k * p
        # for i = 1:p
        #     s = sum(abs2, @view(M[:, i]))
        #     for j = 1:k
        #         rj.Hd[offset + i + (j - 1) * p] = s
        #     end
        # end
        return dot(r, r) / 2
    end
end

function calculate_whbatchsize(nwhbatch, nwratio, p)
    nnwbatch = Int(round(nwhbatch*nwratio)); nnwrbatch = Int(round(nnwbatch/p)) # number of rows
    nnhbatch = nwhbatch - nnwbatch; nnhcbatch = Int(round(nnhbatch/p)) # number of columns
    nnwrbatch, nnhcbatch
end

function makeminibather(k, p, m, n; sd = 0.1)
    nys = k^2; nynw = nysw = m*p; nynh = nysh = n*p; nwratio = nynw/(nynw+nynh)
    batchys = collect(1:k^2)
    return function(cache::AbstractSLMCache, nbatch)
        r = cache.r; nr = length(r)
        Gysdiag = 1.0 .+ sd .* randn(nys)
        Gysdiag = clamp.(Gysdiag, eps(eltype(Gysdiag)), 2.0)
        copyto!(@view(cache.Gdiag[1:nys]), Gysdiag)
        copyto!(@view(cache.Ddiag[1:nys]), Gysdiag)   # must be matched to Gdiag
        
        if nr == nbatch
            return Base.Slice(eachindex(r))
        end
        nnwhbatch = (nbatch - nys)÷2
        nnwrbatch, nnhcbatch = calculate_whbatchsize(nnwhbatch, nwratio, p)
        batchynwr = randperm(m)[1:nnwrbatch]; batchynhc = randperm(n)[1:nnhcbatch]
        batchynw = map(ridx -> nys+ridx:m:nys+ridx+(p-1)*m, batchynwr)
        batchynh = map(cidx -> nys+nynw+p*(cidx-1)+1:nys+nynw+p*cidx, batchynhc)
        batchysw = map(rrng -> rrng.+(nynw+nynh), batchynw)
        batchysh = map(crng -> crng.+(nynw+nynh), batchynh)
        batchy = vcat(batchys, collect.(batchynw)..., collect.(batchynh)..., collect.(batchysw)..., collect.(batchysh)...)

        return  batchy
    end
end

#============ for sparse coding ===========#
function makescop(M::AbstractMatrix{T}, N::AbstractMatrix{T}, Vt::AbstractMatrix{T}, Gh::AbstractMatrix{T}) where T
    k″, n′ = size(Vt); r′, p′ = size(M); p″, r″ = size(N)
    @assert (k″ == r″) && (r′ == r″) && (p′ == p″)
    return LinearOperator{T}(r′*r″+n′*p″, (r′ + r″) * p′, false, false,
            function(res, v, α, β)    # Jacobian-vector product
                dM = reshape(v[1:r′*p′], r′, p′)
                dN = reshape(v[r′*p′+1:end], p″, r″)
                Ys = reshape(view(res,1:r′*r″), r′, r″)
                mul!(Ys, dM, N, α, β)
                mul!(Ys, M, dN, α, true)
                Ysh = reshape(view(res,r′*r″+1:r′*r″+n′*p′), p′, n′)
                mul!(Ysh, dN, Vt, α, β); Ysh .*= Gh  # dYsh=Gh⨀(dN*Vt)
                return res
            end,
            function(res, v, α, β)  # Jacobian-transpose-vector product
                dYs = reshape(v[1:r′*r″], r′, r″)
                dYsh = reshape(v[r′*r″+1:r′*r″+n′*p′], p′, n′)
                dM = reshape(view(res,1:r′*p′), r′, p′)
                dN = reshape(@view(res[r′*p′+1:end]), p″, r″)
                mul!(dM, dYs, N', α, β); mul!(dN, M', dYs, α, β)
                dYsh .*= Gh; mul!(dN, dYsh, Vt', α, true)  # M'*dYs + (Gh⨀dYsh)*Vt'
                return res
            end,
            nothing)
end

# function makescop(M::AbstractMatrix{T}, N::AbstractMatrix{T}, VU::AbstractMatrix{T}, Gw::AbstractMatrix{T}) where T
#     k′, n′ = size(Vt); r′, p′ = size(M); p″, r″ = size(N)
#     @assert (k′ == r′) && (r′ == r″) && (p′ == p″)
#     return LinearOperator{T}(r′*r″+m′*p′, (r′ + r″) * p′, false, false,
#             function(res, v, α, β)    # Jacobian-vector product
#                 dM = reshape(v[1:r′*p′], r′, p′)
#                 dN = reshape(v[r′*p′+1:end], p″, r″)
#                 Ys = reshape(view(res,1:r′*r″), r′, r″)
#                 mul!(Ys, dM, N, α, β)
#                 mul!(Ys, M, dN, α, true)
#                 Ysw = reshape(view(res,r′*r″+1:r′*r″+m′*p′), m′, p′)
#                 mul!(Ysw, U, dM, α, β); Ysw .*= Gw # dYsw=Gw⨀(W*dM)
#                 return res
#             end,
#             function(res, v, α, β)  # Jacobian-transpose-vector product
#                 dYs = reshape(v[1:r′*r″], r′, r″)
#                 dYsw = reshape(v[r′*r″+1:r′*r″+m′*p′], m′, p′)
#                 dM = reshape(view(res,1:r′*p′), r′, p′)
#                 dN = reshape(@view(res[r′*p′+1:end]), p″, r″)
#                 mul!(dM, dYs, N', α, β); dYsw .*= Gw; mul!(dM, U', dYsw, α, true)  # dYs*N' + U'*(Gw⨀dYsw)
#                 mul!(dN, M', dYs, α, β)
#                 return res
#             end,
#             nothing)
# end

g(X::AbstractMatrix, σ) = (X.^2 .+ σ^2).^(1/4)
dg(X::AbstractMatrix, σ) = (0.5*(X.^2 .+ σ^2).^(-3/4)).*X

function make_sc_fcache_op!(M::AbstractMatrix{T}, N::AbstractMatrix{T}, Vt::AbstractMatrix{T},
                        D::AbstractMatrix{T}, Gh::AbstractMatrix{T}, rtαh, σh) where T
    p = size(M,2); k, n = size(Vt)
    x = [vec(D); zeros(n*p)]
    xpred = copy(x)
    Dpred = reshape(view(xpred,1:k^2),k,k)
    Yshpred = reshape(view(xpred,k^2+1:k^2+n*p),p,n)
    Mscratch, Nscratch, xpredscratch = copy(M), copy(N), copy(x)
    Dpredscratch = reshape(view(xpredscratch,1:k^2),k,k)
    Yshpredscrach = reshape(view(xpredscratch,k^2+1:k^2+n*p),p,n)
    return function(rj, θ, idx)
        if rj === nothing
            # Don't modify M, N, Dpred
            copyto!(Mscratch, view(θ, 1:k*p))
            copyto!(Nscratch, @view(θ[k*p+1:end]))
            mul!(Dpredscratch, Mscratch, Nscratch)
            mul!(Yshpredscrach, Nscratch, Vt)
            copyto!(Yshpredscrach, g(Yshpredscrach, σh)*rtαh)
            return sum((x[i] - xpredscratch[i])^2 for i in idx) / 2
        end
        copyto!(M, view(θ, 1:k*p))
        copyto!(N, @view(θ[k*p+1:end]))
        mul!(Dpred, M, N); mul!(Yshpred, N, Vt)
        copyto!(Gh, dg(Yshpred, σh)*rtαh); copyto!(Yshpred, g(Yshpred, σh)*rtαh)
        @inbounds r = x[idx] - xpred[idx]
        rj.r[idx] = r
        return dot(r, r) / 2
    end
end

# function make_sc_fcache_op!(M::AbstractMatrix{T}, N::AbstractMatrix{T}, U::AbstractMatrix{T},
#                         D::AbstractMatrix{T}, Gw::AbstractMatrix{T}, rtαw, σw) where T
#     p = size(M,2); m, k = size(U)
#     x = [vec(D); zeros(m*p)]
#     xpred = copy(x)
#     Dpred = reshape(view(xpred,1:k^2),k,k)
#     Yswpred = reshape(view(xpred,k^2+1:k^2+m*p),m,p)
#     Mscratch, Nscratch, xpredscratch = copy(M), copy(N), copy(x)
#     Dpredscratch = reshape(view(xpredscratch,1:k^2),k,k)
#     Yswpredscrach = reshape(view(xpredscratch,k^2+1:k^2+m*p),m,p)
#     return function(rj, θ, idx)
#         if rj === nothing
#             # Don't modify M, N, Dpred
#             copyto!(Mscratch, view(θ, 1:k*p))
#             copyto!(Nscratch, @view(θ[k*p+1:end]))
#             mul!(Dpredscratch, Mscratch, Nscratch)
#             mul!(Yswpredscrach, U, Mscratch)
#             copyto!(Yswpredscrach, g(Yswpredscrach, σw)*rtαw)
#             return sum((x[i] - xpredscratch[i])^2 for i in idx) / 2
#         end
#         copyto!(M, view(θ, 1:k*p))
#         copyto!(N, @view(θ[k*p+1:end]))
#         mul!(Dpred, M, N)
#         mul!(Yswpred, U, M)
#         copyto!(Gw, dg(Yswpred, σw)*rtαw); copyto!(Yswpred, g(Yswpred, σw)*rtαw)
#         @inbounds r = x[idx] - xpred[idx]
#         rj.r[idx] = r
#         return dot(r, r) / 2
#     end
# end

function calculate_sc_hbatchsize(nshbatch, p)
    Int(round(nshbatch/p)) # number of columns
end

function makescminibather(k, p, n; sd = 0.1)
    nys = k^2; nysh = n*p
    batchys = collect(1:k^2)
    return function(cache::AbstractSLMCache, nbatch)
        r = cache.r; nr = length(r)
        Gysdiag = 1.0 .+ sd .* randn(nys)
        Gysdiag = clamp.(Gysdiag, eps(eltype(Gysdiag)), 2.0)
        copyto!(@view(cache.Gdiag[1:nys]), Gysdiag)
        copyto!(@view(cache.Ddiag[1:nys]), Gysdiag)   # must be matched to Gdiag

        if nr == nbatch
            return Base.Slice(eachindex(r))
        end
        nshbatch = nbatch - nys
        nshcbatch = calculate_sc_hbatchsize(nshbatch, p)
        batchyshc = randperm(n)[1:nshcbatch]
        batchysh = map(cidx -> nys+p*(cidx-1)+1:nys+p*cidx, batchyshc)
        batchy = vcat(batchys, collect.(batchysh)...)

        return  batchy
    end
end

# function makescminibather(k, p, m; sd = 0.1)
#     nys = k^2; nysw = m*p
#     batchys = collect(1:k^2)
#     return function(cache::AbstractSLMCache, nbatch)
#         r = cache.r; nr = length(r)
#         if nr == nbatch
#             return Base.Slice(eachindex(r))
#         end
#         nswbatch = nbatch - nys; nswrbatch = Int(round(nswbatch/p)) # number of rows
#         batchyswr = randperm(m)[1:nswrbatch]
#         batchysw = map(ridx -> nys+ridx:m:nys+ridx+(p-1)*m, batchyswr)
#         batchy = vcat(batchys, collect.(batchysw)...)

#         Gysdiag = 1.0 .+ sd .* randn(nys)
#         Gysdiag = clamp.(Gysdiag, eps(eltype(Gysdiag)), 2.0)
#         copyto!(@view(cache.Gdiag[1:nys]), Gysdiag)
#         copyto!(@view(cache.Ddiag[1:nys]), Gysdiag)   # must be matched to Gdiag

#         return  batchy
#     end
# end

#============ for sparse coding 2 ===========#
function makesc2op(M::AbstractMatrix{T}, N::AbstractMatrix{T}, Vt::AbstractMatrix{T},
                    Gh::AbstractMatrix{T}, Ph::AbstractMatrix{T}, nM::Ref{Float64}) where T
    k″, n′ = size(Vt); r′, p′ = size(M); p″, r″ = size(N)

    @assert (k″ == r″) && (r′ == r″) && (p′ == p″)
    return LinearOperator{T}(r′*r″+n′*p″, (r′ + r″) * p′, false, false,
            function(res, v, α, β)    # Jacobian-vector product
                dM = reshape(v[1:r′*p′], r′, p′)
                dN = reshape(v[r′*p′+1:end], p″, r″)
                Ys = reshape(view(res,1:r′*r″), r′, r″)
                nM1o2 = sqrt(nM[]); nMm3o2o2 = 0.5/nM1o2^3
                mul!(Ys, dM, N, α, β)
                mul!(Ys, M, dN, α, true)
                Ysh = reshape(view(res,r′*r″+1:r′*r″+n′*p′), p′, n′)
                mul!(Ysh, nM1o2*dN, Vt, α, β); Ysh .*= Gh  # dYshn=∥M∥¹ᐟ²Gh⨀(dN*Vt)
                nMm3o2o2dotMdM = nMm3o2o2*dot(M, dM)
                mul!(Ysh, nMm3o2o2dotMdM, Ph, α, true)  # dYshm=1/2*∥M∥⁻³ᐟ²<M,dM>Ph
                return res
            end,
            function(res, v, α, β)  # Jacobian-transpose-vector product
                dYs = reshape(v[1:r′*r″], r′, r″)
                dYsh = reshape(v[r′*r″+1:r′*r″+n′*p′], p′, n′)
                dM = reshape(view(res,1:r′*p′), r′, p′)
                dN = reshape(@view(res[r′*p′+1:end]), p″, r″)
                mul!(dM, dYs, N', α, β); mul!(dN, M', dYs, α, β)
                nM1o2 = sqrt(nM[]); nMm3o2o2 = 0.5/nM1o2^3
                dotPhYsh = dot(Ph, dYsh); mul!(dM, nMm3o2o2*dotPhYsh, M, α, true) # dYs*N' + 1/2*∥M∥⁻³ᐟ²<Ph,dYsh>M
                dYsh .*= Gh*nM1o2; mul!(dN, dYsh, Vt', α, true)  # M'*dYs + ∥M∥¹ᐟ²(Gh⨀dYsh)*Vt'
                return res
            end,
            nothing)
end

function make_sc2_fcache_op!(M::AbstractMatrix{T}, N::AbstractMatrix{T}, Vt::AbstractMatrix{T},
                        D::AbstractMatrix{T}, Gh::AbstractMatrix{T}, Ph::AbstractMatrix{T}, nM::Ref{Float64}, 
                        rtαh, σh) where T
    p = size(M,2); k, n = size(Vt)
    x = [vec(D); zeros(n*p)]
    xpred = copy(x)
    Dpred = reshape(view(xpred,1:k^2),k,k)
    Yshpred = reshape(view(xpred,k^2+1:k^2+n*p),p,n)
    Mscratch, Nscratch, xpredscratch = copy(M), copy(N), copy(x)
    Dpredscratch = reshape(view(xpredscratch,1:k^2),k,k)
    Yshpredscrach = reshape(view(xpredscratch,k^2+1:k^2+n*p),p,n)
    return function(rj, θ, idx)
        if rj === nothing
            # Don't modify M, N, Dpred
            copyto!(Mscratch, view(θ, 1:k*p))
            copyto!(Nscratch, @view(θ[k*p+1:end]))
            mul!(Dpredscratch, Mscratch, Nscratch)
            mul!(Yshpredscrach, Nscratch, Vt)
            copyto!(Yshpredscrach, g(Yshpredscrach, σh)*rtαh*sqrt(norm(Mscratch)))
            return sum((x[i] - xpredscratch[i])^2 for i in idx) / 2
        end
        copyto!(M, view(θ, 1:k*p))
        copyto!(N, @view(θ[k*p+1:end]))
        mul!(Dpred, M, N); mul!(Yshpred, N, Vt)
        copyto!(Gh, dg(Yshpred, σh)*rtαh); copyto!(Ph, g(Yshpred, σh)*rtαh)
        nM[] = norm(M)
        copyto!(Yshpred, Ph*sqrt(nM[]))
        @inbounds r = x[idx] - xpred[idx]
        rj.r[idx] = r
        return dot(r, r) / 2
    end
end
