using Distributions

function wald_test(;Sigma, b, Terms = nothing, L = nothing, H0 = nothing, df = nothing, print_result = true)
    if Terms == nothing && L == nothing
        error("One of the arguments Terms or L must be used.")
    end
    if Terms != nothing && L != nothing
        error("Only one of the arguments Terms or L must be used.")
    end
    if Terms == nothing
        w = size(L)[1]
        Terms = (1:length(b))[(sum(L, dims = 1) .> 0)[1, :]]
    else
        w = length(Terms)
    end
    if H0 == nothing
        H0 = fill(0, w)
    end
    if w != length(H0)
        error("Vectors of tested coefficients and of null hypothesis have different lengths\n")
    end
    if L == nothing
        L = fill(0, w, length(b))
        for i in 1:w
            L[i, Terms[i]] = 1
        end
    end
    f = L * b
    V = Sigma
    mat = inv(L * V * L')
    statistic = (f - H0)' * mat * (f - H0)
    p = ccdf(Chisq(w), statistic)
    if df == nothing
        res = Dict("chi2" => (chi2 = statistic, df = w, P = p))
    else
        fstat = statistic/size(L)[1]
        df1 = size(L)[1]
        df2 = df
        res = Dict("chi2" => (chi2 = statistic, df = w, P = p),
            "Ftest" => (Fstat = fstat, df1 = df1, df2 = df2, P = ccdf(FDist(df1, df2), fstat)))
    end

    if print_result == true
        println("Wald test:\n", "----------\n\n", "Chi-squared test:\n",
            "X2 = ", res["chi2"].chi2, ", df = ", res["chi2"].df, ", P(> X2) = ", res["chi2"].P)
        if df != nothing
            println("\nF test:\n",
            "W = ", res["Ftest"].Fstat, ", df1 = ", res["Ftest"].df1, ", df2 = ", res["Ftest"].df2, ", P(> W) = ", res["Ftest"].P)
        end
    end

    return (Sigma = Sigma, b = b, Terms = Terms, H0 = H0, L = L, result = res, df = df)
end
