.. Tech-Rank algorithm documentation master file, created by
   sphinx-quickstart on Wed Jul  7 14:53:47 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tech-Rank algorithm's documentation
===============================================

Goal
########
TechRank aims to help decision-makers to quantitatively assess the influence of the entities in order to  take good investments decisions under high level of uncertainty. 

Abstract
############
The cybersecurity technological landscape is a complex ecosystem in which entities -- such as  companies and technologies -- influence each other in a non-trivial manner. Understanding influence measures of each entity is central when it comes to  take informed technological  investment  decisions. 

To recognize the mutual influence of companies and technologies in cybersecurity, we consider a bi-partite graph that links companies and technologies. Then, we weight nodes by applying a recursive algorithm based on the method of reflection. This endeavour helps to assign a measure of how an entity impacts the cybersecurity market. Our results help (i) to measure the magnitude of influence of each entity, (ii) decision-makers to address more informed investment strategies, according to their preferences. 

Investors can customzse the algorithm by indicating which external factors --such as previous investments and geographical positions-- are relevant for them. They can select their interests among a list of properties about companies and technologies and weights them according to their needs. This preferences are automatically included in the algorithm and the TechRank's scores changes accordingly.


Code
########

The code is a mix of notebooks and `py` files: 

* the *py files* contain the functions needed in the notebook and the declaration of the classes;

* the *notebooks* explain all the steps.

**Data**:
`Crunchbase (CB) pro <https://www.crunchbase.com/home>`_

**Classes** :
We work with 3 dataclasses: `Companies`, `Technologies` and `Investors`.

**Documentation**:
We create the documentation in HTML using [Sphinx](https://www.sphinx-doc.org/en/master/).


Files description
####################

.. csv-table:: Short description of the files
   :header: "File name ", "Short Description"
   :widths: 15, 30

   "classes.py", "C, T and I dataclasses declaration"
   "create_dictionaries(1).ipynb", "Creation of the Cs and Ts classes, saved as  dictionaries (c_name:class_c and t_name:class_t)"
   "investments_graph(2).ipynb", "Creation of the Is classes, saved as dictionary (i_name:class_i)"
   "main(3).ipynb", "TechRank algorithm (both parameters' optimization and random walker)"
   "plots(4).ipynb", "Bi-partite network plots "
   "create_useful_dataset.ipynb", "Creation .CSV where TechRank results are saved "
   "analysis_results.ipynb", "Analysis of the TechRank results"
   "analysis_investors.ipynb", "Analysis of the investors on CB "
   "crunchbase_api.ipynb", "How make queries and extract data using the CB API"   
   "country_distance.ipynb", "Extraction Cs position and calculation distance from Is"
   "to_run.py", "Implementation of all the functions needed before running TechRank (data cleaning, creation classes, plots...)"
   "functions/fun.py", "Implementation of all the functions for the random walk step"
   "functions/fun_meth_reflections.py", "Implementation of all the functions for the random walk step"
   "functions/fun_external_factors.py", "Implementation of all the functions for the inclusion of the exogenous factors"
   "docs/...", "Material for creating the documentation"
   "docs/build/html/index.html", "Documentation in HTML"
   "plots/...", "All the plots"
   "savings/...", "Results savings"
   "data/sample_CB_date", "Sample of CB data"
   

Each file contains more details and comments. 

Please note that data (also the classes) are not available because they are released by Crunchbase to the CYD Campus with a proprietary licence that does not allow us to share them. However, we plovide a sample of the data. 



Hints of bibliography
########################
The two main resources of this work are the following:


* `The Building Blocks of Economic Complexity <https://rb.gy/rhrsi2>`_ by César A. Hidalgo and Ricardo Hausmann


* `The Virtuous Circle of Wikipedia: Recursive Measures of Collaboration Structures <https://dl.acm.org/doi/10.1145/2675133.2675286>`_ by Maximilian Klein. Thomas Maillart and John Chuang (`GitHub repository <https://github.com/wazaahhh/wiki_econ_capability>`_)


Table of content
####################
.. toctree::

   function_link


