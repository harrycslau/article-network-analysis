# article-network-analysis

### Effects, benefits (✅), and drawbacks (❌) of different parameters:

| **Parameter**   | **Low Value Effect** | **High Value Effect** |
|----------------|----------------------|----------------------|
| **`top_n`** (Number of words to keep) | ✅ Less clutter, easier to interpret. <br> ❌ May **remove** relevant but less frequent words, leading to **loss of context**. | ✅ Provides **more detail** and thematic depth. <br> ❌ Can make the graph **too dense**, making it harder to find key insights. |
| **`min_weight`** (Minimum co-occurrence strength to keep an edge) | ✅ Captures **all possible connections**, useful for **exploratory analysis**. <br> ❌ Introduces **too much noise**, making weak, meaningless connections appear. | ✅ Highlights **stronger** and more meaningful relationships. <br> ❌ Can **disconnect important nodes**, especially if thresholds are too high. |
| **`min_degree`** (Minimum connections a node must have to stay in the graph) | ✅ Preserves **rare but possibly important terms**. <br> ❌ **Isolated words** make the graph harder to read. | ✅ Creates a **more structured** and readable network. <br> ❌ May **eliminate niche concepts**, reducing depth in the analysis. |


### **Example Settings for Different Use Cases (to be amended)**
| Goal | `top_n` | `min_weight` | `min_degree` |
|------|--------|-------------|-------------|
| **Broad Thematic Analysis (More Detail)** | 200 | 1 | 1 |
| **Balanced Graph (Good Readability & Detail)** | 100 | 3 | 2 |
| **Simplified Network (Only Key Themes)** | 50 | 5 | 3 |
