total_mem() = (size=Int(Sys.total_memory())/1e9; println("$(size) GBytes"))
free_mem() = (size=Int(Sys.free_memory())/1e9; println("$(size) GBytes"))

function symbol()
    println("∥ \\parallel, ∦ \\nparallel, ≤ \\leq, ≥ \\geq, ≐ \\doteq, ≍ \\asymp, ⋈ \\bowtie, ≪ \\ll, ≫ \\gg, ≡ \\equiv,")
    println("⊢ \\vdash, ⊣ \\dashv, ⊂ \\subset, ⊃ \\supset, ≈ \\approx, ∈ \\in, ∋ \\ni, ⊆ \\subseteq, ⊇ \\supseteq, ≅ \\cong,")
    println("⌣ \\smile, ⌢ \\frown, ⊈ \\nsubseteq, ⊉ \\nsupseteq, ≃ \\simeq, ⊨ \\models, ∉ \\notin, ⊏ \\sqsubset, ⊐ \\sqsupset,")
    println("∼ \\sim, ⊥ \\perp, ∣ \\mid, ⊑ \\sqsubseteq, ⊒ \\sqsupseteq, ∝ \\propto, ≺ \\prec, ≻ \\succ, ⪯ \\preceq, ⪰ \\succeq,")
    println("≠ \\neq, ∢ \\sphericalangle, ∡ \\measuredangle, ∴ \\therefore, ∵ \\because")
    println("")
    println("± \\pm, ∓ \\mp, ∩ \\cap, ∪ \\cup, ⊎ \\uplus,")
    println("⊕ \\oplus, ⊗ \\otimes, ⊖ \\ominus, ⊘ \\oslash,  ⊙ \\odot, ◯ \\bigcirc,")
    println("△ \\bigtriangleup, ▽ \\bigtriangledown, × \\times, ÷ \\div,")
    println("⊓ \\sqcap, ⊔ \\sqcup, ◃ \\triangleleft, ▹ \\triangleright,")
    println("∗ \\ast, ⋆ \\star, ∨ \\vee, ∧ \\wedge, † \\dagger, ‡ \\ddagger,")
    println("⋄ \\diamond, ∘ \\circ, ∙ \\bullet, ⋅ \\cdot, ∖ \\setminus, ≀ \\wr, ⨿ \\amalg")
    println("")
    println("∈ \\in,  ∋ \\ni,  ∉ \\notin,  ∩ \\cap, ∪ \\cup, ⊂ \\subset, ⊃ \\supset, , ∅ \\emptyset")
    println("→ \\rightarrow or \\to, ← \\leftarrow or \\gets, ↔ \\leftrightarrow, ⇌ \\rightleftharpoons, ↦ \\mapsto,")
    println("⇒ \\Rightarrow,  ⇐ \\Leftarrow, ⇔ \\Leftrightarro, ⟹ \\implies, ⟸ \\impliedby, ⟺ \\iff,")
    println("∃ \\exists,  ∄ \\nexists, ∀ \\forall, ∧ \\land, ∨ \\lor, ⊤ \\top,  ⊥ \\bot,  ¬ \\neg, ∠ \\angle")
    println("")
    println("⟨ \\langle, ⟩ \\rangle, ↑ \\uparrow, ↓ \\downarrow, ⇑ \\Uparrow, ⇓ \\Downarrow,")
    println("⌈ \\lceil, ⌉ \\rceil, ⌊ \\lfloor, ⌋ \\rfloor")
    println("")
    println("∂ \\partial, ı \\imath, ℜ \\Re, ∇ \\nabla	ℵ \\aleph, ð \\eth, ȷ \\jmath, ℑ \\Im,")
    println("◻ \\Box, ℶ \\beth, ℏ \\hbar, ℓ \\ell, ℘ \\wp, ∞ \\infty, ℷ \\gimel")
end

function greek()
    println("A α: A \\alpha,                    B β: B \\beta,                      Γ γ: \\gamma,   Δ δ: \\delta,")
    println("E ϵ ε : E, \\epsilon \\varepsilon, Z ζ: Z \\zeta,                      H η: H \\eta,   Θ θ ϑ: \\Theta \\theta \\vartheta,")
    println("I ι: I \\iota,                     K κ ϰ: K \\kappa \\varkappa,        Λ λ: \\lambda,  M μ: M \\mu,")
    println("N ν: N \\nu,                       Ξ ξ: \\xi,                          O o: O  o,      Π π ϖ: \\Pi \\pi \\varpi,")
    println("P ρ ϱ: P \\rho \\varrho,           Σ σ ς: \\Sigma \\sigma \\varsigma,  T τ: T  \\tau,  Y υ: Y \\upsilon,")
    println("Φ ϕ φ: \\Phi \\phi \\varphi,       X χ: X \\chi,                       Ψ ψ: \\psi,     Ω ω: \\omega")
end

function mkimgUM(U,M,imgsz)
    ncells = size(U,2)
    UM = U*M
    UMrs = reshape(UM, imgsz..., ncells)
    Urs  = reshape(U, imgsz..., ncells)

    # Prepare for display
    mxabs = max(maximum(abs, UMrs), maximum(abs, Urs))
    fsc = scalesigned(mxabs)
    fcol = colorsigned()

    mappedarray(fcol ∘ fsc, reshape(Urs, Val(2))), mappedarray(fcol ∘ fsc, reshape(UMrs, Val(2)))
end

function imshowUM(U,M,imgsz)
    uimg, umimg = mkimgUM(U,M,imgsz)
    if is_ImageView_available
        imshow(uimg)
        imshow(umimg)
    else
        @warn("ImageView is not available!")
    end
end

function imsaveUM(fname,U,M,imgsz)
    uimg, umimg = mkimgUM(U,M,imgsz)
    Images.save("uimg.png", uimg)
    Images.save(fname, umimg)
end

function mkimgW(W::Matrix{T},imgsz; gridcols=size(W,2), borderwidth=1, borderval=0.7, scalemtd=:maxwhole,
        colors=(colorant"green1", colorant"white", colorant"magenta")) where T
    ncells = size(W,2)
    if scalemtd == :maxwhole
        mxabs = max(eps(eltype(W)),maximum(abs, W))
        fsc = scalesigned(mxabs)
        fcol = colorsigned(colors...)
        Wcolor = Array(mappedarray(fcol ∘ fsc, reshape(W, Val(2))))
    elseif scalemtd == :maxcol
        fsc = (x) -> (mxabs=max(eps(eltype(x)),maximum(abs, x)); x./mxabs)
        sW = similar(W)
        for (i,x) in enumerate(eachcol(W)) sW[:,i] = fsc(x) end
        fcol = colorsigned(colors...)
        Wcolor = Array(mappedarray(fcol, reshape(sW, Val(2))))
    elseif scalemtd == :maxgridrow
        fsc = (x) -> (mxabs=max(eps(eltype(x)),maximum(abs, x)); x./mxabs)
        sW = similar(W)
        for i in 1:gridcols:ncells
            x = view(W,:,i:min(i+4,ncells))
            sW[:,i:min(i+4,ncells)]=fsc(x)
        end
        fcol = colorsigned(colors...)
        Wcolor = Array(mappedarray(fcol, reshape(sW, Val(2))))
    elseif scalemtd == :avgwhole
        avgabs = sum(abs,W)/length(W)*18; sW = W./avgabs; clamp!(sW,-1,1)
        fcol = colorsigned(colors...)
        Wcolor = Array(mappedarray(fcol, reshape(sW, Val(2))))
    end
    gridsz = ((ncells-1)÷gridcols+1,gridcols)
    add_dim_sz = ntuple(i->1,Val(length(imgsz)-length(gridsz)))
    bordersz = ntuple(i->borderwidth,Val(2))
    bimgsz = imgsz.+(bordersz..., add_dim_sz...)
    gimgsz = bimgsz.*(gridsz..., add_dim_sz...).+bordersz
    fill_val = eltype(Wcolor)(borderval)
    Wrs = fill(fill_val, gimgsz...)
    for i in 1:ncells
        gi = (i-1)÷gridsz[2]+1
        gj = i-(gi-1)*gridsz[2]
        gindices = (gi-1,gj-1, (add_dim_sz.-add_dim_sz)...)
        offset = gindices.*bimgsz .+ bordersz
        rngs = ntuple(i->offset[i]+1:offset[i]+imgsz[i], length(imgsz))
        Wrs[rngs...] = reshape(Wcolor[:,i], imgsz...)
    end
    Wrs
end

function mkimgH(H::Matrix{T}, tlength=size(H,2); colors=(colorant"green1", colorant"white", colorant"magenta")) where T
    mxabs = maximum(abs, H)
    fsc = scalesigned(mxabs)
    fcol = colorsigned(colors...)
    Array(mappedarray(fcol ∘ fsc, reshape(H[:,1:tlength], Val(2))))
end
