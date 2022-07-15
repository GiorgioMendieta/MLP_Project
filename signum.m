function out = signum(v)

if(v)>0
    out = 1;
elseif(v)== 0
    out = 0;
elseif(v)<0
    out = -1;
end

return