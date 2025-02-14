module clustering

using Clustering, CSV, DataFrames, Dates, Plots

train_df = CSV.read("./data/train_users_2.csv", DataFrame)
#testt_df = CSV.read("./data/test_users.csv", DataFrame)

function processdataframe!(x)
    select!(x, Not(:id))
    dropmissing!(x)
    filter!(row -> all(value -> value != "-unknown-", row), x)
    x[!, names(x, Date)] = (col -> datetime2unix(DateTime(col))).(x[!, names(x, Date)])
    for name in names(x, AbstractString)
        for (i, value) in enumerate(unique(x[!, name]))
            x[!, name] = (string -> string == value ? i : string).(x[!, name])
        end
    end
end

println(describe(train_df))
processdataframe!(train_df)
println(describe(train_df))

x_train_matrix = Matrix(train_df)

result = kmeans(x_train_matrix, 3)
heatmap(x_train_matrix, c=result.assignments)
savefig("output/kmeans.svg")

end
