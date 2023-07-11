function I = MInformation(spikeTrain, N, phi_bin)

    flatphi = phi_bin(:);
    spikeTrain = spikeTrain(:);

    cPhi = histcounts(flatphi, N); % Convert data into probabilities
    cSpike = histcounts(spikeTrain, N);
    cPS = histcounts2(flatphi, spikeTrain, N);
    Pp = cPhi / sum(cPhi);
    Ps = cSpike / sum(cSpike);
    Pps = cPS / sum(cPS(:));

    Hp = entropy(Pp);
    Hs = entropy(Ps);
    Hps = entropy(Pps);
    I = Hp + Hs - Hps;
end