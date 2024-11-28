# NextConvGeN
**Parameters of FDC and NextConvGeN algorithms**

The algorithms discussed in this section, the FDC and the NextConvGeN algorithm, employ specific parameters to configure their behavior and customize their operations. To facilitate the reproducibility of our work, here we specify some details of our experiments.

**FDC algorithm parameters:**
The FDC algorithm requires three main parameters to create an instance of the library: `cont_list`, `nom_list`, and `ord_list`.

* `cont_list`: A list of continuous feature column names. In the context of the FDC algorithm, continuous features refer to variables with numerical values that can take any real number.
* `nom_list`: A list of nominal feature column names. Nominal features represent categorical variables with discrete values with no particular order or hierarchy.
* `ord_list`: A list of ordinal feature column names. Ordinal features are similar to nominal ones but have a specific order or hierarchy.

It should be noted that all other parameters of the FDC algorithm have default values predefined within the library, i.e., they have predetermined values unless otherwise provided.

**NextConvGeN algorithm parameters:**
The NextConvGeN algorithm utilizes five parameters: `neb`, `gen`, `disc_train_count`, `fdc`, and `alpha_clip`.

* `neb`: Determines the number of neighboring data points for generating synthetic points in the convex space.
* `gen`: Determines the number of generated synthetic points the generator creates simultaneously. These synthetic points are generated through convex combinations of the real samples chosen from a neighborhood.
* `disc_train_count`: Indicates the number of additional training steps for the discriminator before updating the generator once. The discriminator is a component of the algorithm that helps assess the quality and authenticity of the generated synthetic points.
* `fdc`: Requires the class instance of FDC to be passed, as it is responsible for searching the neighborhoods that serve as input for the Generator.
* `alpha_clip`: Takes a value between 0 and 1 representing the percentage of clipping the maximum convex coefficient.

We have set the `gen` value equal to `neb` to generate synthetic points that match the size of the input neighborhood.
