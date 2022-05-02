import scipy.stats as stats
import numpy as np

def analyse_stats(var1,var2, paired = False):
    # Test if var1 and var2 are normally distributed
    v1_norm = stats.shapiro(var1)[1]>0.05
    v2_norm = stats.shapiro(var2)[1]>0.05
    # st.write(var1)
    # st.write(var2)
    # test variance homogeneity
    if v1_norm and v2_norm:
        v1_v2_hom = stats.bartlett(var1,var2)[1]>0.05
    else:
        v1_v2_hom = stats.levene(var1,var2)[1]>0.05

    _compare = var1 == var2
    if not isinstance(_compare, bool):
        _compare = _compare.all()

    if _compare:
        result = {
            "v1_norm": v1_norm,
            "v2_norm": v2_norm,
            "v1_v2_hom": v1_v2_hom,
            "v1_mean": var1.mean(),
            "v2_mean": var2.mean(),
            "v1_std": var1.std(),
            "v2_std": var2.std(),
            "v1_median": np.median(var1),
            "v2_median": np.median(var2),
            "p_value": 1
        }
        return result


    if v1_norm and v2_norm:
        if paired:
            result = stats.ttest_rel(var1, var2)
        else:
            result = stats.ttest_ind(var1,var2, equal_var=v1_v2_hom)
    else:
        if paired:
            result = stats.wilcoxon(var1, var2)
        else:
            result = stats.ranksums(var1,var2)

    # st.write(result)
    result = {
        "v1_norm": v1_norm,
        "v2_norm": v2_norm,
        "v1_v2_hom": v1_v2_hom,
        "v1_mean": var1.mean(),
        "v2_mean": var2.mean(),
        "v1_std": var1.std(),
        "v2_std": var2.std(),
        "v1_median": np.median(var1),
        "v2_median": np.median(var2),
        "p_value": result.pvalue,
    }

    return result