"""
X = noisefilter(filter,X)
filter : :medT, :meanT, :medS
"""
function noisefilter(filter,X)
    if filter == :medT
        X = mapwindow(median!, X, (1,3)) # just for each row
    elseif filter == :meanT
        X = mapwindow(mean, X, (1,3)) # just for each row
    elseif filter == :medS
        rsimg = reshape(X,imgsz...,lengthT)
        rsimgm = mapwindow(median!, rsimg, (3,3,1))
        X = reshape(rsimgm,*(imgsz...),lengthT)
    end
    X
end

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