module svm

using CSV, DataFrames, Dates, LIBSVM, MLJ
import MLJBase.predict

SVC = @load SVC pkg=LIBSVM
model = SVC()

train_df = CSV.read("./data/train_users_2.csv", DataFrame)
prdct_df = CSV.read("./data/test_users.csv", DataFrame)
sessn_df = CSV.read("./data/sessions.csv", DataFrame)
unique_strings = Dict()

function processdataframe!(x, y)
    dropmissing!(x)
    filter!(row -> all(value -> value != "-unknown-", row), x)
    x[!, names(x, Date)] = (value -> datetime2unix(DateTime(value))).(x[!, names(x, Date)])
    for name in names(x, AbstractString)
        for value in unique(x[!, name])
            if ! haskey(y, value)
                y[value] = length(y) + 1
            end
            key = y[value]
            x[!, name] = (string -> string == value ? key : string).(x[!, name])
        end
    end
end

function processmissing!(x)
    mapcols!(col -> replace!(col, "-unknown-" => "-NONE-"), x)
    x[!, names(x, Union{Missing, Date})] = (value -> ismissing(value) ? Date(1970, 1, 1) : value).(x[!, names(x, Union{Missing, Date})])
    x[!, names(x, Union{Missing, AbstractString})] = (value -> ismissing(value) ? "-NONE-" : value).(x[!, names(x, Union{Missing, AbstractString})])
    x[!, names(x, Union{Missing, Float64})] = (value -> ismissing(value) ? -1.0 : value).(x[!, names(x, Union{Missing, Float64})])
    mapcols!(col -> replace!(col, missing => -1), x)
end

#println(describe(train_df))
#processed_train_df = copy(train_df)
#select!(processed_train_df, Not(:id))
#select!(processed_train_df, Not(:date_first_booking))
#processdataframe_keepmissing!(processed_train_df, unique_strings)
#println(describe(processed_train_df))
#processed_sessn_df = copy(sessn_df)
#select!(processed_sessn_df, Not(:user_id))
#processdataframe_keepmissing!(processed_sessn_df, unique_strings)

processed_train_df = copy(train_df)
processmissing!(processed_train_df)
processed_sessn_df = copy(sessn_df)
processmissing!(processed_sessn_df)
processed_final_df = leftjoin(processed_train_df, processed_sessn_df, on=[:id => :user_id])
select!(processed_final_df, Not(:id))
select!(processed_final_df, Not(:date_first_booking))
processmissing!(processed_final_df)
println(describe(processed_final_df))
processdataframe!(processed_final_df, unique_strings)
println(describe(processed_final_df))

matrix_size = nrow(processed_final_df) ÷ 2
x_train = select(processed_final_df, Not(:country_destination))[1:matrix_size, :]
x_testt = select(processed_final_df, Not(:country_destination))[matrix_size + 1:end, :]
y_train = categorical(processed_final_df[1:matrix_size, :country_destination])
y_testt = categorical(processed_final_df[matrix_size + 1:end, :country_destination])

mach = machine(model, x_train, y_train, scitype_check_level=0) |> MLJ.fit!
println(accuracy(predict(mach, x_testt), y_testt))

#full_processed_train_df = copy(train_df)
#full_processed_prdct_df = copy(prdct_df)
#processdataframe_keepmissing!(full_processed_train_df, unique_strings)
#processdataframe_keepmissing!(full_processed_prdct_df, unique_strings)

#full_x_train = select(full_processed_train_df, Not(:country_destination))
#full_y_train = categorical(full_processed_train_df[!, :country_destination])
#full_mach = machine(model, full_x_train, full_y_train, scitype_check_level=0) |> MLJ.fit!

#println(predict(full_mach, full_processed_prdct_df))

end
