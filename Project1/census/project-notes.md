
# Identifying trends in Scotland's 2011 census using Data Science and Machine Learning

## Aim

Over 5 million people participated in the last census in Scotland in 2011. Use the census data warehouse to uncover insights into population health, education and travel patterns and apply machine learning to predict a given trait from the underlying census data in each area of Scotland.

For this project complete the following steps:

**Step 1 - Data Exploration and Extraction**

- Inspect the Scotland census data warehouse and select some features of interest for further study
- Extract the data in a format suitable for data analysis

**Step 2 - Interpretation and Visualisation**

- Identify some interesting insights and emerging patterns in your chosen features
- Use visualisation techniques (e.g. histograms, line charts, heatmaps) to aid interpretability of the underlying data

**Step 3 - Machine learning prediction**

- Apply machine learning to predict a target feature of your choosing
- Consider optimisations on your chosen machine learning technique, evaluate performance and comment on any limitations

**Step 4 - Conclusions**

- Elaborate on your findings and discuss the results in a wider context

Please submit your work **as a jupyter notebook** containing all your code and commentary to Learn by **9am on Moday 28th January**.

## Setup

Please read the following notes **before** you start working on the project.

### Project Data

All the census data is hosted locally [here](https://www2.ph.ed.ac.uk/~awashbro/PDAML) and contains the following files and folders:

- `source/`: The entire bulk data files of the 2011 Census for Scotland (also available from the [census website](https://www.scotlandscensus.gov.uk/ods-web/data-warehouse.html#bulkdatatab)
- `lookup/`: Table reference data (see the *Census Data Extraction* section)
- `maps/`: Shape data for geographical regions (see the *Choropleth Map* section)

It is recommended that you download the files under `lookup/` and `maps/` to the machine(s) you are using for your analysis. You **do not** need to download the census source data (~1.3GB) as this will fetched remotely via a request to one the utility methods provided.

### Python Environment

You will be using a provided Python 3 **miniconda** environment. This environment is already installed and configured on the Lab PCs. To initiate the environment just run the following command from the terminal: `module load conda/3.6/pdaml` and launch `jupyter-notebook`

### Directory Structure

Ensure you create and maintain the following directory structure in your work area:

```
census-UUN/
  census-UUN.ipynb
  primer.ipynb
PDAML/
  census/
    census_utils.py
lookup/
maps/
```

where:

- `census-UUN/`: directory containing your jupyter notebook (`census-UUN.ipynb`) and any additional supporting files (e.g code or data) - where "UUN" is replaced with your individual UUN. This is the directory you should submit for marking.
- `primer.ipynb`: example notebook showcasing census utilities (see sections below).
- `PDAML/`: PDAML git repository
- `lookup/`: Table reference data files downloaded from the [local census web area](https://www2.ph.ed.ac.uk/~awashbro/PDAML/census)
- `maps/`: Shape data for geographical regions downloaded from the [local census web area](https://www2.ph.ed.ac.uk/~awashbro/PDAML/census)

Keeping to this structure will help in resolving any setup issues you may encounter also make it easier to fetch any updates to code and instructions in the repository without affecting your work.

The file `primer.ipynb` will need copying from your instance of the PDAML repository into the `census-UUN` folder before it can be used.

### Using your Laptop

All the libraries and the utilities code have been validated on the Lab PCs. You are welcome to use your laptop to complete the project **but we cannot offer you any support if you encounter any setup issues**. The use of Lab PCs to complete the project is therefore **strongly preferred**.

If you have anaconda (or miniconda) available on your laptop then you can try the following:

```bash
# create env
conda create -y -n pdaml-env python=3

# activate env
conda activate pdaml-env

# install packages
conda install -y pandas matplotlib scipy scikit-learn pydotplus seaborn tensorflow ipython ipykernel jupyter jupyterlab
conda install -y folium geojson -c conda-forge
ipython kernel install --user --name=pdaml-env

# deactivate
conda deactivate
```

Then for each session:

```bash
conda activate pdaml-env
jupyter-notebook
```

### Further Help

Please email Dr. Christos Leonidopoulos if you encounter any problems with the project setup and raise any issues well before the deadline date.

## Report Format

The project report has to be in the jupyter notebook format. It is expected that you are comfortable with writing, running and annotating code within a jupyter notebook.

### Annotation and Commentary

It is important that **all** code is annotated and that you provide brief commentary **at each step** to explain your approach. We expect well-documented jupyter notebooks, not an unordered collection of code snippets. For example, explain why you selected the features and the objectives of your study. You can also include any failed approaches if you provide reasonable explanation.

This is not in the form of a written report so do not provide pages of background material. Only provide substantial text for the conclusions step. Aim to clearly present your work so that the markers can easily follow your reasoning and can reproduce each of your steps through your analysis.

To add commentary above (or below) a code snippet create a new cell and add your text in *markdown* format. **Do not** add commentary as a code comment in the same cell as the code. To change the new cell into **markdown** select from the drop down menu on the bar above the main window (the default is *code*)

### Submission Steps

Before marking we will open your notebook and run all the cells so it is important your code is fully functional before it is submitted or this will affect your final mark.

When you are ready to submit your report perform the following steps:

- In Jupyter run `Kernel > Restart Kernel and Run All Cells` to ensure that all your analysis is reproducible and all output can be regenerated
- Run `Kernel > Restart Kernel and Clear All Outputs`
- Save the notebook
- Close Jupyter
- Tar and zip your project folder (i.e. `tar cvfz census-UUN.tar.gz census-UUN/`)
- Submit this file through Learn (specific instructions will follow)

You are free to include any supporting code or data. Make sure this belongs in the `census-UUN` directory and is referenced correctly in your notebook. If your compressed project folder **exceeds 40 MB** please contact Dr. Christos Leonidopoulos ahead of the submission deadline for instructions.

## Utility methods

Several utility methods have been provided to help with your data analysis. You are free to use, modify or use a different set of methods for your study. These are not guaranteed to be bug free so please feel free to flag any obvious problems so fixes can be applied and distributed to all students.

Some of the methods will be familiar from the examples and checkpoint exercises earlier in the course but the interfaces may have changed. The purpose of the additional methods provided will be described in more detail below.

To include the `census_utils` module in your notebook include the line `import census_utils` at the top of your notebook and copy `census_utils.py` from `PDAML/Project1/` to `census-UUN/`.

## Census Data Extraction

The Scotland Census data is available at the [Scotland Census data explorer](https://www.scotlandscensus.gov.uk/ods-web/home.html). You are welcome to download and manipulate this data directly but to save you (huge amounts of) time this data has already been pre-processed and made available via the `extractdata` method.

The segmentation of the census data requires some orientation before you dive into your analysis.

Firstly the data is provided as a number of different *geographies* (i.e. Scottish Parliamentary constituencies, Health Boards, Output Area) which partition Scotland into different areas. For this analysis we will use the *Local Characteristic (LC) Postcode Sector* geography which consists of 1,012 different areas. The choice is somewhat arbitrary (essentially we'd like to have a large enough sample size to perform machine learning classification) and you are free to use a different geography type if you wish. The geography types available are:

- Civil Parish
- Community Health Care Partnership
- Council Area
- DC Postcode Sector
- Health Board
- LC Postcode Sector
- Output Area
- Scottish Parliamentary Constituency
- Scottish Parliamentary Region
- SNS Data Zone 2011
- UK Parliament Constituency

More details on the Census geographies are [here](https://www.scotlandscensus.gov.uk/census-geographies).

All areas are encoded in the following format: `SXXYYYYYY` where `XX` is the geography type (e.g 29 = LC) and `YYYYYY` is the area ID.

The census data is presented as a number of tables covering some attribute (e.g. Highest level of qualification by age). Columns in each table then segment this attribute by some condition (or *category*) (e.g. All people aged 16 and over:_Total_White: Scottish) and column entries show the number of people satisfying that condition in a given area.

Census data accessed via the `extractdata` method is encoded in the following format: `XXYYYYSCZZZZ`
where `XXYYYY` is the table ID (XX is usually DC or LC) and `SCZZZZ` is the category ID.

There are over **60,000** unique categories to choose from. To make choosing a selection of features easier load in the **lookup table** which provides all the table and category information in the form of a dataframe. I suggest that you pick features from across different themes (e.g Health, Education, Language, Work Travel) rather than many from the same table unless you have a specific study in mind. Aim to select **at least 5 different features** for ongoing study and Interpretation.

See the primer notebook for examples on how to lookup and extract features derived from the census data.

## Normalisation Step

In some of the census geographies the areas will have very different populations. Therefore for any meaningful comparison across areas you will need to **normalise** the number of entries against some parent feature.

For example, the *ratio of homeowners to the total number of homes* in an area is a more useful metric than the *total amount of homeowners* in an area. Normalisation must therefore be calculated for each chosen feature as part of your data preparation. The `extractdata` method will help with this step if you provide the set of parent features.

An example of this is shown in the primer notebook.

## Data Visualisation

It is completely up to you how to best illustrate your selected data. Here are some ideas to start with:

- The distribution shape of each of the features
- Identification of any outliers
- Simple correlations or clustering between pairs (or triplets) of features

Consider how you have used `matplotlib` (and `seaborn`) to generate plots earlier in the course - and feel free to experiment with new types of plot if it aids understanding of the underlying data.

Please remember we are looking for *quality* not quantity. Do not provide hundreds of plots if no additional insight can be derived.

## Choropleth Maps

A utility method has been provided to plot your selected data over an interactive map of Scotland. Variations between sectors are highlighted with a colour palette allowing any regional variations to be easily identified. These are known as [choropleth maps](https://en.wikipedia.org/wiki/Choropleth_map)

There are a few libraries in Python that allow you to generate choropleth maps. For this study `folium` was chosen as this allows zooming to explore detail (e.g. in cities) as well as the use of Openstreetmap tiling to display place names underneath your chosen shading.

The utility function `genchoropleth` makes it simple for you to generate these maps or you can call the `folium` package directly. See the primer notebook to see how this works in practce.

The source map data files provided by Scotland Census are too high precision for `folium` so coordinate precision reduction and shape smoothing was applied to keep the file size small (if you zoom in on the generated map you'll see that the regions are not an exact fit to the underlying map - but will be good enough for your purposes). Please note that `folium` struggles rendering choropleth maps for `geojson` files larger than 2MB so the `LC` map is at the limit of what can be used. You'll likely see a small lag when zooming in/out of the map but if you are patient (!) it will be fine to use.

You are welcome to choose an alternative projection package (e.g. cartopy) and your own shapefiles/geojson sources. Be aware - this was painful to pre-process and may not be the best use of your project time!

## Machine learning approaches

Unlike previous machine learning studies in the course it is for you to determine which feature should be considered as the target. As an example (shown in the primer notebook) I have used *General Health* as a target feature and derived a single value metric from the underlying data. Explore the source data to see if you can do something similar. Here are some example features you could consider:

- National Statistics Socio-economic Classification (NS-Sec)
- Distance travelled to work
- Method of Travel
- Industry
- Occupation
- Country of Birth
- Marital and civil partnership status
- Type of Central Heating
- Living Arrangements

Try and choose a target feature that has a distribution with notable variance (i.e. broad peak about some mean value) or maps to a set of ranges without significant class imbalance.

You can re-use the features you selected earlier on in the study or you can extract a new set of features you feel will be good predictors of the target feature. When choosing your target you will have to ensure that you do not introduce [model leakage](https://towardsdatascience.com/data-leakage-part-i-think-you-have-a-great-machine-learning-model-think-again-ad44921fbf34) by choosing predictor features that are too similar to or proxies of the target data.

Note that the emphasis of the marking will be on the approach you take and *not* on your final model performance. You should be able to explain why your model performed poorly if you did not get a satisfactory result (conversely you should also be able to explain why your model was good at explaining your target feature).

## Transformation into a Classification Study

The majority of the data is unlabelled and is therefore initially suited to a regression study (i.e. predicting the *value* of the target in a given area).

You can either perform a regression study using appropriate machine learning methods or *segment* the target data into a number of classes using the `genclfscore` and `genclfsingle` methods. Classes can defined either as a number of distinct bands of a single category or bands of some combined score derived from multiple features.

For more details see the classification example in the primer notebook.

## Conclusions

Use the closing section of your project to elaborate on your findings and place your study in a wider context. Here are some ideas to consider:

- Did your study prove your disprove your initial beliefs?
- Did your study draw out any interesting insights that were not obvious from the outset?
- What were the limitations of the dataset?
- What future work do you suggest if you were looking to perform a more extensive study?
- What additional public datasets could be included as part of a future study?
- Are there any policies or actions (e.g social planning, regional infrastructure) that would be informed by or benefit from your results?

Think "big" but make your assertions evidence-based and free of any personal bias - and please avoid making any controversial statements particularly on gender, religion or ethnicity.

## Any Questions?

This is a new study with large scope and as such there may be setup issues that have not been considered. Please send any questions and feedback as early as possible to give us time to address them well before the deadline. We will keep you notified if there are any significant changes need to be made.

Last of all have fun exploring the data! We look forward to reading your findings!
