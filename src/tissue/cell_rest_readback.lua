-- Passive transverse rest-length readback: the actual rest of each central
-- transverse bond (columns nu/2-1 and nu/2, 0-based) as created, per sheet row.
-- Read-only; corrects metrology that assumed a uniform H*(1-pretension) rest on
-- non-uniform central cells. Loaded after the coupon create override so T exists
-- when called; never alters tissue state.
function cellTransverseRestLengths()
 local out={}
 for j=0,T.nv-1 do local a=j*T.nu+12; local b=a+1; local rest=nil
  for _,s in ipairs(T.springs) do if (s.a==a and s.b==b) or (s.a==b and s.b==a) then assert(not rest); rest=s.rest end end
  assert(rest and rest>0);out[#out+1]=rest
 end
 return out
end
