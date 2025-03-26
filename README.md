# article-network-analysis

### **Quick Guide to Interpreting the Graph**  

✅ **Colors** → Each color represents a **thematic group (cluster)** of words that frequently appear together. Words in the same color are more **strongly related** in meaning.  

✅ **Lines (Edges)** → A line (connection) means **two words co-occur** within the same part of the manuscript.  
- **Thicker lines** = Stronger relationships (words appear together often).  
- **Thinner lines** = Weaker connections (words rarely appear together).  

✅ **Node Size** → Bigger words are **more important** in the text.  
- **High-degree words** (many connections) are **key discussion points**.  
- **Smaller words** may be **supporting concepts**.  

✅ **Clusters (Groups of Nodes)** → Densely connected areas indicate **core topics** in the manuscript. Isolated clusters may suggest **separate discussions** that are not well integrated.  


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
