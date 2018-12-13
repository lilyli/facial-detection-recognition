def fun(deg):
    mag = 10
    orient_bins = [0, 20, 40, 60, 80, 100]
    hist = [0] * 5
    ind = np.digitize(deg, orient_bins) - 1
    # split magnitude between two closest bins
    if deg % (orient_bins[1] - orient_bins[0]) == 0:
    # direction falls perfectly between two bins
        adj_ind = -1
    elif deg % ((orient_bins[1] - orient_bins[0]) / 2) == 0:
    # direction falls perfectly in the center of a bin
        adj_ind = -2
    elif deg > (orient_bins[ind + 1] + orient_bins[ind]) / 2:
        if ind == len(hist) - 1:
            # wrap around to first bin
            adj_ind = 0
        else:
            adj_ind = ind + 1
    elif deg < (orient_bins[ind + 1] + orient_bins[ind]) / 2:
        if ind == 0:
            # wrap around to last bin
            adj_ind = len(hist) - 1
        else:
            adj_ind = ind - 1

    print(ind, adj_ind)

    try:
        if adj_ind == -1:
            pct_split = 0.5
            hist[ind] += pct_split * mag
            if ind == 0:
                a_ind = len(hist) - 1
            elif ind == len(hist) - 1:
                a_ind = 0
            else:
                a_ind = ind - 1
            hist[a_ind] += pct_split * mag
            print(pct_split)
        elif adj_ind == -2:
            hist[ind] += mag
        else:
            if (adj_ind < ind) or (adj_ind == len(hist) - 1 and ind == 0): # account for case where bin wraps around end
                pct_split = np.abs((orient_bins[ind + 1] - deg) / (orient_bins[1] - orient_bins[0]))
            else:
                pct_split = np.abs((orient_bins[ind] - deg) / (orient_bins[1] - orient_bins[0]))
            hist[ind] += pct_split * mag
            hist[adj_ind] += (1 - pct_split) * mag
            print(pct_split)
    except Exception as e:
        print(e)

    print([10, 30, 50, 70, 90])
    print(hist)