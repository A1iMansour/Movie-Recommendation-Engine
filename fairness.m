function fair = fairness(Um,Gg,Uf,userIDs_m,moviesinG, userIDs_f,U,V,p,q)
    fair1=1/(Um*Gg)*(U(userIDs_m, :) * V(moviesinG, :)' + p(user_id));
    fair2=;

    fair=(fair1-fair2)^ 2;

    
end

