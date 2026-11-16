# A small JSON reader and writer, enough for the simulator's protocol
# (objects, arrays, strings without unicode escapes beyond \uXXXX, numbers,
# true, false, null).

module MiniJSON

export parse_json, to_json

function skipws(s, i)
    while i <= lastindex(s) && isspace(s[i])
        i = nextind(s, i)
    end
    i
end

function parse_value(s, i)
    i = skipws(s, i)
    c = s[i]
    if c == '{'
        d = Dict{String,Any}()
        i = skipws(s, i + 1)
        s[i] == '}' && return d, i + 1
        while true
            k, i = parse_value(s, i)
            i = skipws(s, i)
            s[i] == ':' || error("expected ':' at $i")
            v, i = parse_value(s, i + 1)
            d[k] = v
            i = skipws(s, i)
            if s[i] == ','
                i += 1
            elseif s[i] == '}'
                return d, i + 1
            else
                error("expected ',' or '}' at $i")
            end
        end
    elseif c == '['
        a = Any[]
        i = skipws(s, i + 1)
        s[i] == ']' && return a, i + 1
        while true
            v, i = parse_value(s, i)
            push!(a, v)
            i = skipws(s, i)
            if s[i] == ','
                i += 1
            elseif s[i] == ']'
                return a, i + 1
            else
                error("expected ',' or ']' at $i")
            end
        end
    elseif c == '"'
        io = IOBuffer()
        i += 1
        while true
            ch = s[i]
            if ch == '"'
                return String(take!(io)), i + 1
            elseif ch == '\\'
                e = s[i+1]
                if e == 'n'
                    write(io, '\n')
                elseif e == 't'
                    write(io, '\t')
                elseif e == 'u'
                    write(io, Char(parse(Int, s[i+2:i+5], base = 16)))
                    i += 4
                else
                    write(io, e)
                end
                i += 2
            else
                write(io, ch)
                i = nextind(s, i)
            end
        end
    elseif startswith(SubString(s, i), "true")
        return true, i + 4
    elseif startswith(SubString(s, i), "false")
        return false, i + 5
    elseif startswith(SubString(s, i), "null")
        return nothing, i + 4
    else
        j = i
        while j <= lastindex(s) && (isdigit(s[j]) || s[j] in "+-.eE")
            j += 1
        end
        t = s[i:j-1]
        v = occursin(r"[.eE]", t) ? parse(Float64, t) : parse(Int, t)
        return v, j
    end
end

parse_json(s::AbstractString) = parse_value(s, 1)[1]

to_json(x::Nothing) = "null"
to_json(x::Bool) = x ? "true" : "false"
to_json(x::Integer) = string(x)
to_json(x::AbstractFloat) = isfinite(x) ? string(Float64(x)) : "null"
to_json(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n") * "\""
to_json(x::Symbol) = to_json(String(x))
to_json(x::AbstractVector) = "[" * join(map(to_json, x), ", ") * "]"
to_json(x::Tuple) = to_json(collect(x))
to_json(x::AbstractDict) = "{" * join(["$(to_json(string(k))): $(to_json(v))" for (k, v) in x], ", ") * "}"
to_json(x::NamedTuple) = to_json(Dict(string(k) => v for (k, v) in pairs(x)))

end
