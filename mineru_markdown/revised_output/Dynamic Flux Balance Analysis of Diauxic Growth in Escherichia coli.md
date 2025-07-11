# Dynamic Flux Balance Analysis of Diauxic Growth in Escherichia coli

Radhakrishnan Mahadevan, Jeremy S. Edwards, and Francis J. Doyle, III Department of Chemical Engineering, University of Delaware, Newark, Delaware 19716 USA

ABSTRACT Flux Balance Analysis (FBA) has been used in the past to analyze microbial metabolic networks. Typically, FBA is used to study the metabolic flux at a particular steady state of the system. However, there are many situations where the reprogramming of the metabolic network is important. Therefore, the dynamics of these metabolic networks have to be studied. In this paper, we have extended FBA to account for dynamics and present two different formulations for dynamic FBA. These two approaches were used in the analysis of diauxic growth in Escherichia coli. Dynamic FBA was used to simulate the batch growth of E. coli on glucose, and the predictions were found to qualitatively match experimental data. The dynamic FBA formalism was also used to study the sensitivity to the objective function. It was found that an instantaneous objective function resulted in better predictions than a terminal-type objective function. The constraints that govern the growth at different phases in the batch culture were also identified. Therefore, dynamic FBA provides a framework for analyzing the transience of metabolism due to metabolic reprogramming and for obtaining insights for the design of metabolic networks.

# INTRODUCTION

Recent developments in genomics, such as genome sequencing, microarrays, and GeneChips have provided detailed information into the genetic networks of several microorganisms (De Saizieu et al., 1998; Tao et al., 1999; Selinger et al., 2000; Oh and Liao, 2000; Wei et al., 2001). The next logical step is to use this information to study the integrated behavior of the cellular networks. One of the areas of research has been the study of metabolic networks (Oh and Liao, 2000; Tao et al., 2001; Ideker et al., 2001). The analytical and experimental methods for understanding the nature of flux distribution in a metabolic network, along with molecular biology techniques for genetic engineering, assist in the design of the metabolic reaction networks (Stephanpoulos, 1999). Mathematical analysis of metabolism can guide the metabolic engineering process; for example, Hatzimanikatis et al. (1998) have addressed the problem of determining the optimal regulatory structure in terms of gene overexpression or deletion. In that study, the regulatory structure was represented by binary variables, and the objective was to maximize a desired steady-state objective through the solution of a mixed integer linear programming formulation. Several other quantitative approaches have been proposed to study metabolic networks. These approaches include metabolic control analysis (Fell, 1996), biochemical systems theory (Savageau et al., 1987a,b), cybernetic modeling (Kompala, 1984; Dhurjati et al., 1985; Varner and Ramkrishna, 1999), and flux balance analysis (FBA) (Varma and Palsson, 1994b). With the exception of FBA, these approaches require a functional form for the kinetics of the cellular reactions.

FBA is an approach to constrain the metabolic network based on the stoichiometry of the metabolic reactions and does not require kinetic information (Varma and Palsson, 1994a). Optimization of an objective function, such as growth rate, is used to obtain a metabolic flux distribution that satisfies the constraints, and FBA has been shown to provide meaningful predictions in Escherichia coli (Varma and Palsson, 1994b; Edwards et al., 2001). van Riel and coworkers have proposed a modified FBA approach, where the flux balance analysis problem was solved along with constraints on the rate of change of metabolite levels at specific time instants (van Riel et al., 2000; Giuseppin and van Riel, 2000).

Diauxic growth represents the classical reprogramming of a metabolic network and has been extensively studied with mathematical modeling (Varma and Palsson, 1994b; Wong et al., 1997; Lendenmann and Egli, 1998; Guardia and Calvo, 2001). Ramkrishna and coworkers have also used the cybernetic modeling approach to model the diauxic growth of $E$ . coli on mixtures of glucose and organic acids such as pyruvate, succinate, and fumarate (Ramakrishna et al., 1996; Narang et al., 1997). In cybernetic modeling, the bacterial cell is viewed as an optimal strategist that maximizes the utility of the resources provided to it. The regulation of the genes and the activity of the enzymes are obtained as a solution to an optimal resource allocation problem. Because these variables are obtained as a function of the kinetic rate equations, the result is a closed-form dynamic model of the network.

Herein, we describe dynamic flux balance analysis (DFBA), which incorporates rate of change of flux constraints. We show that DFBA can predict the dynamics of diauxic growth. Classical FBA has also been used to study diauxic growth on glucose and acetate (Varma and Palsson, 1994b). However, classical FBA incorrectly predicted the reutilization of acetate. Furthermore, classical FBA cannot predict the metabolite concentrations, which is possible with DFBA. DFBA also allows the incorporation of kinetic expression when the kinetics are well characterized. In this paper, two different formulations for DFBA are presented.

These two approaches are used to analyze the diauxic growth of $E$ . coli on glucose and acetate. The sensitivity of the approaches to the rate of change of flux constraints, the functional form of the objective function, and the parameters in the model are examined. Using this formalism, we have characterized the different phases of batch growth in terms of the active constraints during each phase. Thus, DFBA provides a significant improvement over the classical FBA and will find utility as a quantitative analysis tool in the basic sciences and biotechnology.

# DYNAMIC FLUX BALANCE ANALYSIS

FBA is a modeling approach that constrains the metabolic network by the balance of the metabolic fluxes (reactions) around each node (metabolite). When the metabolic network is operating in a steady state, the mass balances are described by a set of linear equations,

$$
A \cdot \nu = 0 ,
$$

where $A$ is the $m \times n$ stoichiometric matrix of the reactions, $m$ is the number of the metabolites, $n$ is the number of fluxes, and $\nu$ is the flux vector of the network. Because the system of linear equations is underdetermined (more unknown fluxes than equations), an objective function is used to obtain a solution using linear programming (LP). Typically, the maximization of the growth flux is used as the objective function (Varma and Palsson, 1994a; Bonarius et al., 1997; Pramanik and Keasling, 1997; Edwards et al., 2001), where the growth flux is defined in terms of the biosynthetic requirements. For details on the LP formulation, see Varma and Palsson (1994a), and Edwards et al. (1999). FBA only identifies the metabolic flux distribution, and there is no information on the metabolite concentrations or on the dynamic characteristics of the metabolic fluxes. In simulations where there is a transition between two steady states, the FBA solution will indicate an instantaneous change of the metabolic fluxes (Varma and Palsson, 1994b). Therefore, constraints on the rate of change of the fluxes must be explicitly incorporated in the problem. The dynamic extension to FBA can be formulated in the following two ways. The two formulations are discussed in detail in the following sections.

Dynamic Optimization Approach (DOA): This involves optimization over the entire time period of interest to obtain time profiles of fluxes and metabolite levels. The dynamic optimization problem was transformed to a nonlinear programming (NLP) problem and the NLP problem was solved once. The details of the objective function and the constraints in the formulation are presented in Eq. 3.

Static Optimization Approach (SOA): This approach involves dividing the batch time into several time intervals and solving the instantaneous optimization problem at the beginning of each time interval, followed by integration over the interval. The optimization problem was solved using LP repeatedly during the course of the batch to obtain the flux distribution at a particular time instant. The SOA formulation is presented in detail in Eq. 4. The objective used in the optimization problem can be similar to the objective in FBA. Varma et al. (1994b) have used FBA to obtain dynamic prediction for diauxic growth in a manner similar to this approach. However, they did not incorporate rate-of-change constraints on the metabolic fluxes.

# Dynamic optimization-based DFBA approach

Consider a metabolic network with $m$ metabolites and $n$ fluxes. The set of conservation of mass equations, for each metabolite, results in a set of ordinary differential equations,

$$
{ \frac { \mathrm { d } z } { \mathrm { d } t } } = A \mathbf { v } X , \quad { \frac { \mathrm { d } X } { \mathrm { d } t } } = \mu X , \quad \mu = \Sigma w _ { \mathrm { i } } \nu _ { \mathrm { i } } ,
$$

where $\mathbf { z }$ is the vector of metabolite concentrations, $X$ is the biomass concentration, $\mathbf { V }$ is the vector of metabolic fluxes per gram (DW) of the biomass, $A$ is the stoichiometric matrix of the metabolic network, $\mu$ is the growth rate obtained as a weighted sum of the reactions that synthesize the growth precursors, and $w _ { \mathrm { i } }$ are the amounts of the growth precursors required per gram (DW) of biomass.

Along with the system of dynamic equations, several additional constraints must be imposed for a realistic prediction of the metabolite concentrations and the metabolic fluxes. These include non-negative metabolite and flux levels, limits on the rate of change of fluxes, and any additional nonlinear constraints on the transport fluxes. A general dynamic optimization problem can be formulated as

$$
\begin{array} { r l r } {  { \mathbf { M a x } } } & { \hat { w } _ { \mathrm { e n d } } \Phi ( \mathbf { z } , \mathbf { v } , X ) | _ { \mathrm { t = t } } }  \\ & { } & \\ & { } & { \qquad + \hat { w } _ { \mathrm { i n s } } \sum _ { \mathrm { j = 0 } } ^ { \mathrm { M } } \int _ { \mathrm { t } } ^ { t _ { \mathrm { f } } } L ( \mathbf { z } , \mathbf { v } , X ( t ) ) \delta ( t - t _ { \mathrm { j } } ) \mathrm { d } t } \end{array}
$$

$$
{ \frac { \mathrm { d } \mathbf { z } } { \mathrm { d } t } } = A \mathbf { v } X
$$

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } X } { \mathrm { d } t } = \mu X } \\ { \displaystyle \mu = \Sigma w _ { 1 } \nu _ { i } } \\ { \displaystyle t _ { j } = t _ { 0 } + j \frac { t _ { j } - t _ { 0 } } { M } j = 0 \cdots M } \\ { \displaystyle c ( \mathbf { v } , \mathbf { z } ) \leq 0 \quad \lvert \dot { \mathbf { v } } \rvert \leq \dot { \mathbf { v } } _ { \mathrm { a n ~ } } \quad \forall t \in [ t _ { 0 } , t _ { \mathrm { d } } ] } \\ { \displaystyle z \geq 0 \quad X \geq 0 \quad \forall t \in [ t _ { 0 } , t _ { \mathrm { d } } ] } \\ { \displaystyle z ( t _ { 0 } ) = z _ { 0 } \quad X ( t _ { 0 } ) = X _ { 0 } , } \end{array}
$$

where $\mathbf { z } _ { 0 }$ and $X _ { 0 }$ are the initial conditions for the metabolite concentration and the biomass concentration, respectively, $c ( \mathbf { v } ,$ z) is a vector function representing nonlinear constraints that could arise due to consideration of kinetic expressions for fluxes, $t _ { 0 }$ and $t _ { \mathrm { f } }$ are the initial and the final times, $\Phi$ is the terminal objective function that depends on the end-point concentration, $L$ is the instantaneous objective function,  is the Dirac-delta function, $t _ { \mathrm { j } }$ is the time instant at which $L$ is considered, $\hat { w } _ { \mathrm { i n s } }$ and $\hat { w } _ { \mathrm { e n d } }$ are the weights associated with the instantaneous and the terminal objective function, respectively, and $\mathbf { v } ( t )$ is the time profile of the metabolic fluxes. If the nonlinear constraint is absent, the problem reduces to an optimization involving a bilinear system.

The dynamic optimization problem was solved by parameterizing the dynamic equations through the use of orthogonal collocation on finite elements (Cuthrell and Biegler, 1987). The time period $( t _ { 0 } { - } t _ { \mathrm { f } } )$ was divided into a finite number of intervals (finite elements). The fluxes, the metabolite levels, and the biomass concentration were parameterized at the roots of an orthogonal polynomial within each finite element. The details of the parameterization for a specific example are presented in the next section. Continuity of the metabolite and the biomass concentrations was imposed at the beginning of each of the finite elements. The time derivative of the variables was approximated as a linear combination of the value of the fluxes at each point, and the dynamic equations were transformed to algebraic equations. The nonlinear constraints were imposed at discrete points in the time interval considered. Thus the dynamic optimization was converted to an NLP problem. The resulting NLP was solved using the fmincon function in MATLAB (The MathWorks Inc., Natwick, MA).

were constant over the interval. The optimization problem was then formulated at the next time instant and solved. This procedure was repeated from $t _ { 0 }$ to $t _ { \mathrm { f } } .$ For the class of systems involving only bilinear terms with fluxes and the biomass concentration, it is possible to directly solve the dynamic equations and thereby eliminate the numerical integration.

# DIAUXIC GROWTH OF E. COLI ON GLUCOSE AND ACETATE

The metabolic network considered for modeling the diauxic growth of E. coli is shown in Fig. 1. From a metabolic pathway analysis with glucose, acetate, and oxygen as the input and biomass and acetate as the output, a set of $\sim 3 0 0$ extreme pathways were identified (Schilling et al., 2000a). The biomass composition and the ratio of precursors required were obtained from the literature (Schilling et al., 2000b). From this set, four pathways were chosen based on the biomass yield and the known physiology of $E .$ coli (Cronan and Laporte, 1996; Oh and Liao, 2000) to define a simplified metabolic network (see Figs. 2 and 3). The extreme pathways chosen represented both aerobic and anaerobic utilization of glucose and had the highest biomass yield from among the 300 pathways. The acetate utilization pathway was chosen to be consistent with experimental observations that the pckA gene coding for the PEP carboxykinase is expressed during growth on acetate (Oh and Liao, 2000). The simplified network was then used in all further studies presented in the paper.

A dynamic model for the prediction of the time profiles for a batch bioreactor with glucose as the carbon source is presented in the equations,

# Static optimization-based DFBA approach

In SOA, the time period was divided into $N$ intervals. In the absence of the nonlinear constraints involving the fluxes, the optimization problem is reduced to an LP problem. The LP was solved at the beginning of each interval to obtain the fluxes at that time instant:

$$
\begin{array} { c c } { \mathrm { M a x ~ } \sum w _ { \mathrm { i } } \nu _ { \mathrm { i } } ( t ) } & \\ { \mathrm { v ( t ) } } & { } \\ { \mathbf { z } ( t + \Delta T ) \geq 0 } & { \mathbf { v } ( t ) \geq 0 } \\ { \bar { c } ( \mathbf { z } ( t ) ) \mathbf { v } ( t ) \leq 0 } & { \forall t \in [ t _ { 0 } , t _ { \mathrm { f } } ] } \\ { | \mathbf { v } ( t ) - \nu ( t - \Delta T ) | \leq \dot { \mathbf { v } } _ { \mathrm { m a x } } \Delta T } & { \forall t \in [ t _ { 0 } , t _ { \mathrm { f } } ] } \\ { \mathbf { z } ( t + \Delta T ) = \mathbf { z } ( t ) + A \mathbf { v } \Delta T } \\ { X ( t + \Delta T ) = X ( t ) + \mu X ( t ) \Delta T , } \end{array}
$$

$$
\begin{array} { l } { \displaystyle \frac { G l c x t } { \mathrm { d } t } = { \cal A } ^ { \mathrm { G r x } } \nu { \cal X } , } \\ { \displaystyle } \\ { \displaystyle \frac { \mathrm { d } { \cal A } c } { \mathrm { d } t } = { \cal A } ^ { \mathrm { \tiny ~ A } } \nu { \cal X } , } \\ { \displaystyle \frac { \mathrm { d } { \cal O } _ { 2 } } { \mathrm { d } t } = { \cal A } ^ { \mathrm { \tiny ~ O } _ { 2 } } \nu { \cal X } + k _ { \mathrm { L } } a ( 0 . 2 1 - 0 _ { 2 } ) , } \\ { \displaystyle } \\ { \displaystyle \frac { \mathrm { d } { \cal X } } { \mathrm { d } t } = ( \nu _ { 1 } + \nu _ { 2 } + \nu _ { 3 } + \nu _ { 4 } ) { \cal X } , } \end{array}
$$

where $\Delta T$ is the length of the time interval chosen.

The optimization problem was solved using CPLEX. The dynamic equations were integrated assuming that the fluxes where $\boldsymbol { A } ^ { \mathrm { G l c x t } } , \boldsymbol { A } ^ { \mathrm { A c } } , \boldsymbol { A } ^ { \mathrm { O _ { 2 } } }$ are the rows of the stoichiometric matrix associated with glucose, acetate, and oxygen, respectively, $k _ { \mathrm { L } } a$ is the mass transfer coefficient for oxygen and is assumed to be $7 . 5 \ \mathrm { h r } ^ { - 1 }$ (Edwards et al., 2001).

The key variables in the mathematical model of the metabolic network are the glucose concentration, the acetate concentration, the biomass concentration, and the oxygen concentration in the gas phase. The oxygen concentration in the gas phase was assumed to be a constant $( 0 . 2 1 \mathrm { m M }$ ). A term for the oxygen transport from the gas phase (air at ambient temperature) was included in the model. The oxygen transport rate was assumed to be directly proportional to the difference in concentration. The oxygen uptake rate was constrained to allow a maximum possible flux of 15 mmol/gdw hr (Varma and Palsson, 1994b). Transport of acetate across the cell was assumed to be rapid (with respect to the metabolic flux); therefore, the internal and the external concentrations were assumed to be the same. The glucose uptake rate was bounded by Michaelis– Menten kinetics involving the glucose concentration (Wong et al., 1997). The DFBA formulation for the analysis of diauxic growth in $E$ . coli is presented in the next subsection.

![](images/09dbbefe5322e4e9ab6ecfeb8c00fe7bc475686606fdc52938b45e704bdc0613.jpg) [Image Description: The image is a biochemical pathway diagram representing the metabolic processes in a biological system, likely a cell. This type of diagram is commonly used in biochemistry and molecular biology to illustrate the sequence of chemical reactions within an organism's cells.

The diagram includes various chemical compounds and enzymes involved in the metabolic pathways. The compounds are represented by three-letter abbreviations, and the enzymes are represented by their respective names or abbreviations. Arrows indicate the direction of the reactions, showing how one compound is converted into another by the action of an enzyme.

Key components of the diagram include:

1. **Glucose Metabolism**: Starting with "Glcxt" (extracellular glucose), the pathway shows the conversion of glucose to glucose 6-phosphate (G6P) and further to fructose 6-phosphate (F6P) and dihydroxyacetone phosphate (DHAP). These intermediates are part of glycolysis, which breaks down glucose to produce pyruvate (PYR).

2. **Pentose Phosphate Pathway**: An alternative pathway from G6P leads to the production of ribose 5-phosphate (R5P) and sedoheptulose 7-phosphate (S7P), which are important for nucleotide synthesis.

3. **Citric Acid Cycle (TCA Cycle)**: Pyruvate is converted to acetyl-CoA (AcCoA), which enters the TCA cycle. The cycle includes several steps, producing citrate (CIT), isocitrate (ICIT), alpha-ketoglutarate (AKG), succinyl-CoA (SuccCoA), and other intermediates, ultimately leading to the production of carbon dioxide (CO2) and reduced cofactors NADH and FADH2.

4. **Electron Transport Chain**: The reduced cofactors NADH and FADH2 are used in the electron transport chain, which generates ATP through oxidative phosphorylation. The diagram shows the flow of electrons through complexes I-IV, leading to the production of water (H2O) and ATP.

5. **Anaplerotic Reactions**: The diagram also includes anaplerotic reactions, which replenish TCA cycle intermediates. These include the conversion of pyruvate to oxaloacetate (OAA) and the synthesis of malate (MAL) and aspartate (ASP).

6. **Bypass Pathways**: There are bypass pathways such as the conversion of pyruvate to ethanol (ETH) and acetate (AC), which can occur under certain conditions.

The diagram illustrates the interconnectedness of various metabolic pathways and the flow of energy and reducing equivalents through the cell. It supports the key conclusion that cells utilize a complex network of reactions to convert nutrients into energy and building blocks necessary for growth and maintenance.]  
FIGURE 1 Metabolic network of $E$ . coli considered for the FBA. The network consisted of 54 metabolites and 85 reactions. Glycolysis, pentose phosphate pathway, TCA cycle with the glyoxylate bypass, anapleurotic reactions, and redox metabolism are included in the metabolic network.

![](images/1e7ae10d0ef55ca297a5bef6d74944e39e642b22e2eae75a81d97bcf6398e109.jpg) [Image Description: The image appears to be a schematic representation of metabolic pathways or chemical reactions, possibly related to cellular respiration or fermentation processes. On the left side, there is a circular diagram with four vectors (V1, V2, V3, V4) pointing towards a central point labeled 'X'. These vectors originate from three different points labeled 'Glcxt', 'O2', and 'Ac', which likely represent glucose, oxygen, and acetate, respectively. The vectors suggest different reaction pathways or fluxes leading to the production or consumption of 'X'.

On the right side, there are four equations corresponding to the vectors V1 through V4. Each equation represents a different stoichiometric relationship between the reactants (Ac, Glcxt, O2) and the product 'X'. The coefficients in front of the reactants indicate the molar ratios in which they are consumed or produced in the reactions leading to 'X'. For instance, V1 shows that 39.43 moles of acetate (Ac) and 35 moles of oxygen (O2) are required to produce 'X'. Similarly, V2 indicates that 9.46 moles of glucose (Glcxt) and 12.92 moles of oxygen (O2) are needed to produce 'X'. V3 and V4 show more complex relationships involving both glucose and acetate, with V4 indicating a negative coefficient for glucose, suggesting consumption rather than production.

The key conclusion supported by this diagram is the depiction of different metabolic pathways or conditions under which the substance 'X' can be formed or utilized, with varying requirements of glucose, oxygen, and acetate. This could be used to analyze metabolic fluxes in a biological system under different environmental conditions or genetic modifications.]  
FIGURE 2 Simplified metabolic network. The network identified after pathway analysis with glucose, acetate, and oxygen as the input and biomass as the output and selection based on biomass yield is presented above.

# DFBA: DOA formulation

The DOA formalism of DFBA was used to analyze the diauxic growth of $E$ . coli. The objective function for the DOA formalism is detailed in the equations,

Case 1: Instantaneous objective

$$
J _ { 1 } ( \mathbf { z } , \mathbf { v } , X ) = \sum _ { \mathrm { i } = 1 } ^ { \mathrm { N } _ { \mathrm { s } } } \frac { X ( i ) } { X _ { 0 } e ^ { \mu ^ { \mathrm { s c } } t _ { \mathrm { i } } } }
$$

![](images/1893581e51c5f17d94808128eabdd70b8fb0b0ecbc229ba4991deb9dbcdd89ca.jpg) [Image Description: The image is a schematic representation of various metabolic pathways, specifically focusing on the interactions between glycolysis, the citric acid cycle (also known as the Krebs cycle or TCA cycle), and oxidative phosphorylation. It appears to be a scientific diagram used to illustrate the biochemical processes that occur within cells to produce energy.

The diagram is divided into four quadrants, each depicting a slightly different version of these metabolic pathways. Each quadrant contains a series of chemical compounds and enzymes, with arrows indicating the direction of the reactions. Key components include:

1. **Glycolysis Pathway**: This is represented at the top left of each quadrant, showing the breakdown of glucose (Glu) into pyruvate (Pyr), producing ATP, NADH, and other intermediates like G6P, F6P, and DHAP.

2. **Citric Acid Cycle**: In the center of each quadrant, the cycle is depicted with compounds such as Acetyl-CoA (AcCoA), oxaloacetate (OAA), and succinyl-CoA (SucCoA). The cycle generates NADH, FADH2, and ATP.

3. **Oxidative Phosphorylation**: This process is shown at the top right of each quadrant, involving the electron transport chain (ETC) and ATP synthase, where NADH and FADH2 produced in the previous steps are used to generate ATP.

4. **Pentose Phosphate Pathway**: Indicated by the presence of compounds like G6P, 6PG, and R5P, this pathway provides NADPH and pentose sugars.

5. **Enzymes and Transporters**: Various enzymes (e.g., HK, PFK, PYK) and transport proteins (e.g., GLUT, MCT) are labeled, showing their roles in catalyzing reactions or transporting molecules across cellular membranes.

6. **Regulatory Points**: Some steps are highlighted with dashed lines or double arrows, indicating points of regulation or multiple possible pathways.

The diagram does not contain numerical data or specific rates of reactions, so it is not a chart in the traditional sense. Instead, it serves as an educational tool to visualize the complex network of biochemical reactions that are crucial for cellular energy metabolism. The key conclusion supported by this diagram is the integrated nature of these pathways and how they work together to maintain cellular energy homeostasis.]  
FIGURE 3 The metabolic pathways used to simplify the network $\nu _ { 1 }$ (top left), $\nu _ { 2 }$ (top right), $\nu _ { 3 }$ (bottom left), and $\nu _ { 4 }$ (bottom right). The details of the pathways in the simplified network are shown above. The active reactions are highlighted reactions in the pathways.

Case 2: Terminal time objective

$$
J _ { 2 } ( \mathbf { z } , \mathbf { v } , X ) = X ( N _ { \mathrm { s } } )
$$

$\boldsymbol { Z } ^ { \mathrm { s t } } \in \mathbf { R } ^ { 4 \mathrm { N _ { s } } }$ is the stacked vector containing the metabolite and biomass concentrations in time; $K _ { \mathrm { m } }$ is the saturation constant ( $\mathrm { 0 . 0 1 5 ~ m M }$ , Wong et al., 1997); $\mathbf { z } _ { 0 }$ is a vector consisting of the initial glucose, acetate, and oxygen concentrations; $N _ { \mathrm { v } }$ is the number of collocation points of the fluxes; $V ^ { \mathrm { s t } } \in \mathbb R ^ { 4 \mathrm N _ { \mathrm { v } } }$ is the stacked vector containing the fluxes in time; $\dot { \mathbf { v } } _ { \mathrm { m a x } }$ is the rate of change of flux constraints imposed; $C ^ { 0 }$ is the matrix containing the derivative weights; $f ( Z ^ { \mathrm { s t } } , \ V ^ { \mathrm { s t } } )$ is the function containing the derivative vector along with the continuity condition (determined from Eq. 5); and $\mu ^ { \mathrm { { s c } } }$ is the growth rate determined from the initial and final biomass concentration measurements used in scaling the objective function.

$$
\begin{array} { r l } & { \mathbf { z } _ { 0 } = \left[ 1 0 . 8 ~ 0 . 4 ~ 0 . 2 1 \right] ^ { \mathrm { T } } } \\ & { X _ { 0 } = 0 . 0 0 1 } \\ & { \dot { \mathbf { v } } _ { \mathrm { m a x } } = \left[ 0 . 1 ~ 0 . 3 ~ 0 . 3 ~ 0 . 1 \right] ^ { \mathrm { T } } } \\ & { d ^ { \mathrm { G i c a t } } \mathbf { v } \leq \frac { 1 0 G l x t } { K _ { \mathrm { m } } + G l c x t } \frac { \mathrm { m m o l } } { { \mathrm { g d w } } \cdot \mathrm { h r } } } \\ & { d ^ { 0 } \mathbf { z } \mathbf { v } \leq 1 5 \frac { \mathrm { m m o l } } { { \mathrm { g d w } } \cdot \mathrm { h r } } , } \end{array}
$$

where $N _ { \mathrm { s } }$ is the number of collocation points for the parameterization of the metabolite and biomass concentrations;

For the DOA formalism, each time interval was divided into five finite elements, and the variables were parameterized at the roots of the fifth-order Legendre polynomial, resulting in 204 variables. The flux rates-of-change constraints were included in the optimization problem as linear constraints. The NLP was solved for two different objective functions involving the biomass concentration, and the results are presented in the next section. The first objective function (instantaneous objective (Eq. 6a) involved maximizing the scaled sum of the biomass concentration at the collocation points. As the biomass concentration increases 1000-fold during the course of the batch, the concentrations at different time points were scaled, so that all the time points are equally weighted. The second objective function (terminal time objective (Eq. 6b) maximized the biomass concentration at the final time.

# DFBA: SOA formulation

For DFBA using SOA, the time of the batch $( 1 0 \ \mathrm { h r s } )$ was divided into 10,000 intervals, and the optimization was formulated as described in the Eq. 4 and was solved using CPLEX. The number of variables in the optimization problem was four (corresponding to the number of the fluxes). The optimization was solved at the beginning of each interval, and the metabolite concentrations at the beginning of the next interval were found by direct integration.

![](images/96cd25de77a87f32548c9a5609f26b9ef5d1fa5a06ae8c5cb90a2a66feceefa4.jpg) [Image Description: The image is a composite of four graphs, each representing different data trends over time, specifically measured in hours (hr). These graphs are likely related to a biological or chemical process, possibly fermentation, given the presence of terms like acetate, biomass, oxygen concentration, and glucose concentration.

1. The top-left graph is a line graph with the y-axis labeled "mmol/(gDW hr)" and the x-axis labeled "Time (hr)". It shows four different lines, each representing a different variable (v1, v2, v3, v4). The lines show varying rates of change over time, with v1 and v2 maintaining a relatively stable level, v3 showing a sharp increase around 5 hours, and v4 showing a gradual increase.

2. The top-right graph is a dual-axis line graph. The left y-axis is labeled "Acetate (mM)" and the right y-axis is labeled "Biomass (g/L)". Both are plotted against the x-axis labeled "Time (hr)". The acetate concentration increases sharply around 5 hours and then decreases, while the biomass increases steadily over time.

3. The bottom-left graph is a line graph with the y-axis labeled "Oxygen conc. (mM)" and the x-axis labeled "Time (hr)". It shows a sharp decrease in oxygen concentration over time, starting at approximately 0.2 mM and approaching zero by 10 hours.

4. The bottom-right graph is a line graph with the y-axis labeled "Glucose conc. (mM)" and the x-axis labeled "Time (hr)". It shows a decrease in glucose concentration over time, starting at approximately 10 mM and approaching zero by 10 hours. The line is labeled "Glcext", which might refer to external glucose concentration.

The key conclusions that can be drawn from these graphs are that the process being studied involves the consumption of glucose and oxygen, the production of acetate and biomass, and that these processes change over time. The sharp changes in acetate and oxygen concentration suggest critical points in the process, possibly indicating shifts in metabolic activity or the onset of anaerobic conditions.]  
FIGURE 4 Model prediction using the SOA for DFBA in the absence of the rate of change of flux constraints. Interpretation of the constraints governing the growth of $E$ . coli in the three phases is shown above. In the first phase, the constraints are the oxygen and the glucose uptake rate. The transport of oxygen along with the glucose uptake constrained the growth in the middle phase. Growth on acetate in the final phase was again constrained by oxygen transport. Glucose, acetate, and biomass concentrations from experimental data are plotted along with the model predictions (Varma and Palsson, 1994b).

The parameters used for the DFBA were the maximal oxygen and the glucose uptake rates (Varma and Palsson, 1994b), the mass transfer coefficient (Edwards et al., 2001), the substrate saturation constant (Wong et al., 1997), and the flux rate-of-change constraints. The only parameter that could not be identified based on the existing literature was the flux rate-of-change constraints. These parameters, however, can be estimated from biochemical parameters such as the transcription and translation rates and genomic information involving regulatory elements, microarray data, and proteomics (Tavazoie et al., 1999; Cohen et al., 2000; Kirkpatrick et al., 2001). Thus, in the case where the transcription and translation rates are known, the rate of change of flux constraints can be identified precisely. For the current study, a range of values for the rate of change of fluxes provided reasonable agreement between the model predictions and the experimentally observed time domain data. A single parameter set within the range was chosen for the present study.

# RESULTS AND DISCUSSION

The DFBA approaches were used to simulate batch growth of $E$ . coli on glucose, where acetate is initially secreted and subsequently utilized. The data from a batch fermentation (Varma and Palsson, 1994b) is also plotted in all the figures.

# Static optimization-based approach: Results

The results from the DFBA using the SOA are shown in Figs. 4 and 5. In Fig. 4, the flux rate-of-change constraints were relaxed for the purpose of comparison. The DFBA solution was used to identify the constraints governing cellular growth. It was determined that different constraints were active during different times in the batch culture. We defined distinct phases of the fermentation based on differences in the active constraint. It was observed that, up to 4.6 hr, the constraints on the oxygen and glucose uptake rates were limiting growth and were the active constraints. In the next phase of the fermentation (from 4.6 to $6 . 9 \ \mathrm { h r } $ ) the oxygen concentration in the fermentation environment approached zero, and the system was constrained by the transport of oxygen (governed by $k _ { \mathrm { L } } a$ term). At $6 . 9 \ \mathrm { h r }$ , the glucose was nearly completely consumed, and, from this point until the glucose concentration reached zero, the system was limited by the glucose (Michaelis–Menten kinetics) and oxygen uptake rate constraints. When the glucose concentration was zero, the acetate utilization began, and the growth was characterized by the oxygen transport limitation, which was influenced by the $k _ { \mathrm { L } } a$ term. The growth in the final phase (on acetate) was linear, and not exponential as in the previous phase, due to the $k _ { \mathrm { L } } a$ constraint.

![](images/2bbc20cb893951aca6c1b28237d2458178f5ad70389f40d834f10273ced49e11.jpg) [Image Description: The image is a composite of four graphs, each depicting different variables over time, measured in hours (hr). The graphs are likely related to a biological or chemical process, possibly fermentation, given the presence of acetate, biomass, oxygen, and glucose.

1. The top-left graph shows a variable measured in mmol/(gdw hr) against time. There are four lines representing different conditions or treatments labeled v1, v2, v3, and v4. The lines show a sharp increase in the variable's value at around 5 hours, with v1 and v2 peaking higher than v3 and v4.

2. The top-right graph displays acetate concentration in mM and biomass in g/L against time. There are two lines: one for acetate concentration (solid line) and one for biomass (square markers). The acetate concentration increases sharply around 5 hours, peaking at around 8 mM, while the biomass increases more gradually, reaching a peak of approximately 0.8 g/L.

3. The bottom-left graph shows oxygen concentration in mM over time. The oxygen concentration decreases steadily from the start, dropping to nearly zero by 10 hours.

4. The bottom-right graph depicts glucose concentration in mM over time. Similar to the oxygen concentration, the glucose concentration decreases steadily, dropping to nearly zero by 10 hours.

The key conclusions that can be drawn from these graphs are that there is a significant change in the measured variables around the 5-hour mark, which could indicate the onset of a critical phase in the process being studied. The decrease in oxygen and glucose concentrations suggests consumption of these substrates over time, which is typical in fermentation processes. The increase in acetate and biomass indicates the production of these byproducts and growth of the biological system, respectively.]  
FIGURE 5 Model prediction using SOA for DFBA in the presence of the rate of change of flux constraints. The constraints governing the growth are similar to the previous figure except for the region where the growth is constrained by the rate of change of flux constraints, and pathway 3 is active. Glucose, acetate, and biomass concentrations from experimental data are plotted along with the model predictions (Varma and Palsson, 1994b).

![](images/45dcdd36e9fb5b81a2939afd8c38dc364e84a989a146e505905833543a8da4f1.jpg) [Image Description: The image displays four separate graphs, each representing the relationship between flux (in mmol/gdw hr) and oxygen uptake rate (in mmol/gdw hr). These graphs are likely from a scientific study examining the effects of oxygen uptake on different fluxes in a biological or chemical system.

1. The top left graph shows Flux (v1) on the y-axis and Oxygen uptake rate on the x-axis. The flux remains at zero until an oxygen uptake rate of approximately 10 mmol/gdw hr, after which it increases linearly.

2. The top right graph depicts Flux (v2) against the same oxygen uptake rate. Flux starts at zero and increases linearly with the oxygen uptake rate until it reaches a plateau at around 15 mmol/gdw hr.

3. The bottom left graph presents Flux (v3), which remains constant at zero across all measured oxygen uptake rates, indicating no change in this flux with varying oxygen uptake.

4. The bottom right graph shows Flux (v4), which decreases linearly as the oxygen uptake rate increases, starting from a positive value and approaching zero as the oxygen uptake rate reaches 20 mmol/gdw hr.

These graphs suggest that different fluxes respond differently to changes in oxygen uptake rate, with some fluxes increasing, one remaining constant, and another decreasing. This could be indicative of various metabolic pathways or reactions within a system that are influenced by oxygen availability.]  
FIGURE 6 Initial flux distribution as a function of the oxygen uptake rate for the SOA to DFBA. When the oxygen uptake rate is not sufficient to support aerobic growth (pathway 2), then the anaerobic pathway $( \nu 4 )$ becomes active.

The flux rate-of-change constraints were also imposed on the metabolic network, and the simulations produced similar results (Fig. 5), with the exception of additional phases that were governed by the flux rate of change constraints. The flux rate-of-change constraints were active from 5.5 to $6 . 5 \ \mathrm { h r }$ , where the flux from pathway 3 that produced both biomass and acetate was nonzero. This was due to the constraint on the flux rate of change of the pathway that produced acetate in the absence of oxygen (pathway 4).

# Sensitivity to the oxygen uptake rate

The flux distribution during the early stages of the batch culture was qualitatively defined by the oxygen uptake rate. The by-product formation for the batch growth of $E$ . coli has previously been shown to depend on the oxygen uptake rate (Varma et al., 1993). Therefore, we investigated the optimal flux distribution during the initial growth phase as a function of the maximum oxygen uptake rate. Figure 6 shows that, as the maximum allowed oxygen uptake was decreased, the flux of pathway 4 that produced acetate increased, and, when the maximum allowed oxygen uptake rate was increased, the flux of pathway 4 decreased to zero, and pathways 1 and 2 were active. However, the flux through pathway 3 (produces both biomass and acetate) was found to be zero for all values of the oxygen uptake rate.

![](images/34af7cfed9f59910fdb78860dbeca118c3c0b52e12fc612f55563bd6e97915ab.jpg) [Image Description: The image is a composite of four graphs, each depicting different variables over time, measured in hours (hr). These graphs are likely related to a biological process, possibly fermentation or a similar metabolic activity.

1. The top-left graph shows a variable measured in mmol/(gDW hr) against time. There are four lines representing different conditions or treatments labeled v1, v2, v3, and v4. The lines show a sharp decrease in the variable's value, reaching a minimum around 5 hours, and then a sharp increase, with v4 showing the highest final value.

2. The top-right graph displays two variables: acetate concentration in mM and biomass in relative units (not specified). Both variables increase over time, with acetate concentration peaking sharply around 5 hours and then decreasing, while biomass continues to increase more gradually.

3. The bottom-left graph shows oxygen concentration in mM over time. The concentration decreases sharply and then levels off, indicating a rapid consumption of oxygen.

4. The bottom-right graph depicts glucose concentration in mM over time. Similar to the oxygen graph, glucose concentration decreases sharply and then levels off, suggesting consumption of glucose.

The key conclusions that can be drawn from these graphs are that the process involves the consumption of oxygen and glucose, the production of acetate, and an increase in biomass. The different lines in the first graph suggest that varying conditions (v1 to v4) affect the rate and extent of these processes. The sharp changes around the 5-hour mark across multiple graphs suggest a critical time point in the process, possibly a shift in metabolic pathways or a response to limiting substrates.]  
FIGURE 7 Model prediction using SOA for DFBA in the presence of the rate of change of flux constraints for a glucose uptake rate of 11 mmol/gdwhr. Insufficient oxygen uptake rate due to the increased glucose uptake results in the shutting down of the acetate utilization pathway in the initial phase. Glucose, acetate, and biomass concentrations from experimental data are plotted along with the model predictions (Varma and Palsson, 1994b).

# Sensitivity to the glucose uptake rate

The DFBA solutions described above were generated with a maximum glucose uptake rate of 10 mmol/gdwhr. We used this value because it has been identified experimentally. The sensitivity of the solution to this flux constraint was examined using the SOA. When the maximum glucose uptake rate was increased to 11 mmol/gdwhr (Fig. 7), it was observed that the acetate utilization pathway was not active during the initial stages of the batch. In this case, the oxygen uptake rate was not sufficient to allow acetate utilization as seen earlier in Fig. 6. These results indicated that glucose and oxygen are not simultaneously consumed due to oxygen uptake constraints. However, if the glucose uptake rate is constraining bacterial growth, acetate and glucose are optimally co-metabolized during the initial phase of growth. However, they are not optimally cometabolized once the biomass reaches a higher level.

# Sensitivity to the mass transfer coefficient $( { \sf k } _ { L } \sf a )$

DFBA was performed for a perturbation in the mass transfer coefficient $( k _ { \mathrm { L } } a = 1 2 . 5 ~ \mathrm { h r } ^ { - 1 } )$ ) (Fig. 8). This perturbation could be interpreted as increasing the agitation rate or increasing the surface area of the gas–liquid interphase. Additionally, a similar effect would be obtained by increasing the concentration of oxygen in the sparging gas. Due to the increased rate of oxygen transport, the time when the oxygen concentration reached zero increased, and the pathways that use oxygen increased in activity relative to the acetate-producing pathway (pathway 4). This resulted in decreased acetate production. Because, in the model, the use of acetate depends on the oxygen transport rate, as the $k _ { \mathrm { L } } a$ increases, the acetate use rate increased and acetate concentration decreased to zero at $8 \ \mathrm { h r }$ compared to $9 . 5 \ \mathrm { h r }$ in Fig. 5 for a case where $k _ { \mathrm { L } } a = 7 . 5 ~ \mathrm { h r } ^ { - 1 }$

![](images/33bb9ad11510da358a4510ca038d958a2ca5cc31667bf6975aba5ccca388d5c5.jpg) [Image Description: The image is a composite of four graphs, each depicting different variables over time, measured in hours (hr). These graphs are likely related to a biological or chemical process, possibly fermentation, given the presence of terms like acetate, biomass, oxygen, and glucose.

1. The top-left graph shows a variable measured in mmol/(gDW hr) plotted against time. There are four different lines representing different versions (v1, v2, v3, v4). The lines show a sharp decrease in the variable's value, reaching a plateau around 5 hours, with v4 showing a slightly different pattern, maintaining a higher value before decreasing.

2. The top-right graph displays two variables: acetate concentration in mM and biomass in g/L, both plotted against time. The acetate concentration increases sharply after about 5 hours, while the biomass shows a peak around the same time before decreasing.

3. The bottom-left graph shows oxygen concentration in mM over time. The oxygen concentration decreases steadily and reaches zero around 5 hours, after which it remains constant.

4. The bottom-right graph depicts glucose concentration in mM over time. Similar to the oxygen graph, the glucose concentration decreases steadily and reaches zero around 5 hours, after which it remains constant.

The key conclusions that can be drawn from these graphs are that the process being studied involves a consumption of oxygen and glucose, which leads to an increase in acetate and biomass production. The different versions (v1 to v4) may represent different experimental conditions or strains, with v4 showing a distinct behavior compared to the others. The sharp changes around the 5-hour mark suggest a critical point in the process, possibly indicating a transition in the metabolic state of the system.]  
FIGURE 8 Model prediction using SOA for DFBA in the presence of the rate of change of flux constraints for the case where $k _ { \mathrm { L } } a = 1 2 . 5 ~ \mathrm { h r } ^ { - 1 }$ . Final phase involving acetate utilization is constrained by the transport of oxygen. Increased oxygen availability results in higher rate of acetate utilization. Glucose, acetate, and biomass concentrations from experimental data are plotted along with the model predictions (Varma and Palsson, 1994b).

# Dynamic optimization based approach: Results

The results from the DOA for DFBA are presented in Fig. 9. The rate of change of flux constraints were imposed at all time instants, unlike SOA (where the constraints were relaxed whenever the concentrations were close to zero). Therefore, for this case, the time evolution was marginally slower than the case of dynamic FBA using SOA for the same parameter set. However, when the flux rate of change constraints were relaxed, time evolution was rapid (Fig. 10). Sensitivity studies similar to the previous approach (SOA) were performed, and the results of the simulations for this approach (DOA) were similar. This is to be expected because these two approaches were formulated to produce the same results. The differences in the two approaches are related to the flexibility in problem formulation and the computational requirements (see Discussion).

![](images/253bcc2ec48ffe5b5d461dd8675687426f41d7b860be3d70db7523fcfe289dd6.jpg) [Image Description: The image is a compilation of four graphs, each representing different data trends over time, measured in hours (hr). These graphs are likely related to a biological or biochemical process, possibly fermentation or cell culture, given the variables measured.

1. The top-left graph is a line graph with the y-axis labeled "mmol/(gDW hr)" and the x-axis labeled "Time (hr)". It shows four different lines representing different versions (v1, v2, v3, v4) of a process or experiment. The lines show varying rates of a certain measurement over time, with v2 and v3 peaking around 5 hours and then decreasing, while v1 and v4 show a more gradual increase and decrease.

2. The top-right graph is a combination of two y-axes and one x-axis. The left y-axis is labeled "Acetate (mM)" and the right y-axis is labeled "Biomass (g/L)". Both are plotted against time on the x-axis. There are two sets of data points: squares for acetate concentration and diamonds for biomass. The acetate concentration increases sharply up to around 8 hours and then levels off, while the biomass increases gradually over the entire time period.

3. The bottom-left graph is a line graph with the y-axis labeled "Oxygen conc. (mM)" and the x-axis labeled "Time (hr)". It shows a single line that decreases sharply from the start, reaching close to zero by around 5 hours, and then remains stable.

4. The bottom-right graph is a line graph with the y-axis labeled "Glucose conc. (mM)" and the x-axis labeled "Time (hr)". It shows a single line that decreases steadily over time, starting at around 10 mM and approaching zero by 10 hours.

The key conclusions that can be drawn from these graphs are that the process being studied involves the consumption of glucose and oxygen, the production of acetate, and an increase in biomass over time. The different versions (v1 to v4) of the process show variations in the rate at which these changes occur.]  
FIGURE 9 Model prediction using DOA for DFBA in the presence of the rate of change of flux constraints. The results shown above are similar to the earlier results obtained using SOA. Glucose, acetate, and biomass concentrations from experimental data are plotted along with the model predictions (Varma and Palsson, 1994b).

![](images/ab0112b5237a1a53c3bab97ff944eef1a39b16c4fcea5b75998d8310d1ec624a.jpg) [Image Description: The image is a compilation of four graphs, each depicting different variables over time, measured in hours (hr). These graphs are likely related to a study of microbial metabolism or bioprocesses.

1. The top-left graph shows a variable labeled "mmol/(gDW hr)" on the y-axis, which could represent a rate of metabolic activity per gram of dry weight per hour. The x-axis is labeled "Time (hr)". There are four different lines representing different conditions or treatments (v1, v2, v3, v4). The lines show varying levels of activity over time, with v2 and v3 showing a peak around 5 hours, while v1 and v4 show a more gradual increase.

2. The top-right graph displays two variables: "Acetate (mM)" and "Biomass (g/L)" on the y-axis. The x-axis is also labeled "Time (hr)". The acetate concentration increases sharply up to around 5 hours and then plateaus, while the biomass increases more gradually, peaking slightly after the acetate and then declining.

3. The bottom-left graph shows "Oxygen conc. (mM)" on the y-axis against "Time (hr)" on the x-axis. The oxygen concentration decreases steadily over time, suggesting consumption by a biological process.

4. The bottom-right graph depicts "Glucose conc. (mM)" on the y-axis over "Time (hr)" on the x-axis. Similar to the oxygen graph, the glucose concentration decreases steadily, indicating its consumption over time.

The key conclusions that can be drawn from these graphs are that the metabolic activity varies with different conditions, acetate and biomass production are linked and peak at different times, and both oxygen and glucose are consumed over the course of the experiment, which is typical in studies of microbial fermentation or respiration processes.]  
FIGURE 10 Model prediction using DOA for DFBA in the absence of the rate of change of flux constraints. Glucose, acetate, and biomass concentrations from experimental data are plotted along with the model predictions (Varma and Palsson, 1994b).

# Sensitivity to the objective function

The DOA formalism provides increased flexibility in the definition of the constraints and the objective function. Namely, because the DOA solves the entire solution (time course) in a single optimization problem, objectives that span multiple time steps can be incorporated. For example, with the DOA, the time-dependent flux distribution that maximizes the biomass at the end of the fermentation was solved. Furthermore, other interesting objective functions can be poised, such as maintaining homeostasis and robustness to perturbations in the environment.

We examined the sensitivity of the results to the objective function. We formulated the maximal growth objective in two distinct manners, maximal biomass at the end of the fermentation and maximal growth rate at each instant. Figure 11 depicts the results for the maximization of the endpoint biomass concentration objective. Here, the results obtained differ markedly from the previous case. The pathway that utilizes glucose was active until the end of the batch, and acetate production was slower, and the end-point biomass concentration achieved was greater than the previous cases. These results do not match the experimental data. The results obtained using the instantaneous objective function are more representative of the experimental data. This indicates that $E$ . coli may lack the predictive capability for redirecting the fluxes that could result in increased endpoint biomass concentration.

![](images/ee6eab465b24009a6913a8b98c57a18dedeb264b9246d23155f21a08a570bbcc.jpg) [Image Description: The image is a composite of four graphs, each depicting different variables over time, measured in hours (hr). These graphs are likely related to a biological or chemical process, possibly fermentation, given the presence of terms like acetate, biomass, oxygen, and glucose.

1. The top-left graph shows a variable measured in mmol/(gDW hr) plotted against time. There are four lines representing different conditions or treatments labeled v1, v2, v3, and v4. The lines show varying trends, with v1 and v2 starting higher and decreasing over time, while v3 and v4 start lower and increase, with v4 showing a more pronounced increase.

2. The top-right graph displays two variables: acetate concentration in mM and biomass in g/L, both plotted against time. The acetate concentration increases over time, particularly after the 5-hour mark, while the biomass shows a more complex pattern with an initial increase followed by a decrease and then a plateau.

3. The bottom-left graph shows oxygen concentration in mM over time. The oxygen concentration decreases sharply over the first few hours and then levels off, indicating a consumption of oxygen over time.

4. The bottom-right graph depicts glucose concentration in mM, labeled as Glc^ext, against time. The glucose concentration decreases steadily over time, suggesting it is being consumed.

The key conclusions that can be drawn from these graphs are that the process involves the consumption of glucose and oxygen, the production of acetate, and changes in biomass over time. The different lines in the first graph suggest that varying conditions (v1 to v4) affect the rate of these processes.]  
FIGURE 11 Model prediction using DOA for DFBA where the objective is maximizing the end-point biomass concentration. The results obtained for this objective function do not match the experimental data. The biomass concentration achieved is higher than the previous case, and pathway 2 is active until the end of the batch. Glucose, acetate, and biomass concentrations from experimental data are plotted along with the model predictions (Varma and Palsson, 1994b).

# Discussion

We have extended the classical FBA for analyzing the dynamic reprogramming of a metabolic network. In particular, we have examined the reprogramming of the metabolic network that occurs at different stages of diauxic growth of E. coli on glucose. Two approaches for DFBA were introduced, and the sensitivity to the different parameters was analyzed. The results were compared to the data presented in Varma and Palsson (1994b).

DFBA using SOA extended the FBA approach presented in Varma and Palsson (1994b) through the incorporation of the flux rate-of-change constraints. In this paper, the model for diauxic growth of $E$ . coli considered the effect of oxygen transport, and the metabolic network studied was simplified using pathway analysis to obtain a compact representation. The scope of the results obtained for modeling the metabolic reprogramming during diauxic growth presented here were similar to those based on FBA. Cybernetic models have also been proposed for the study of diauxic growth (Ramakrishna et al., 1996; Narang et al., 1997). The fluxes in the cybernetic approach are obtained as a solution to an optimal resource allocation problem with an instantaneous objective function. Typically, only a subset of the network is considered in the optimization problem (Varner, 2000). The solution obtained is analytic, and one can represent the system with a dynamic model. However, the cybernetic approach requires kinetic information for all the reactions in the network. DFBA does not require kinetic information and considers the entire network, although the solution for the fluxes is not analytic and is obtained by solving an optimization problem.

DFBA using DOA allows the formulation of a dynamic objective function describing characteristics, such as, reduction of transition time between metabolic states (Torres, 1994) or end-point biomass optimization, into a rigorous mathematical framework. A dynamic objective function based on the desired goal could provide information useful in the design of genetically modified metabolic networks for metabolic engineering by taking into account the dynamic responses to fluctuations in the system. The static optimization-based DFBA would not allow such a dynamic formulation, because the optimization performed is at a specific time instant. However, in SOA, the number of variables that have to be solved is far fewer (in each optimization) in comparison, and the optimization problem is an LP problem as opposed to the NLP for DOA. As the size of the network increases, the number of variables and the number of constraints would increase proportionally in the NLP. Thus, SOA is scalable to larger metabolic networks.

DFBA provides a framework for modeling the dynamic responses of a metabolic network to various perturbations. In this paper, we have examined the applicability of this framework for modeling the diauxic growth in $E$ . coli. The results from DFBA are qualitatively similar to the experimental observations. DFBA was able to predict the onset of acetate production and also the preference of $E$ . coli for sequential utilization of acetate and glucose over the simultaneous utilization. The constraints governing the behavior were identified at various phases in the batch culture. It was found that, in the initial phase, the glucose and oxygen uptake rates were the active constraints. In the middle phase, the oxygen concentration is close to zero, and the mass transfer coefficient $( k _ { \mathrm { L } } a )$ and the maximum allowed rate of change of flux was found to constrain the system. Acetate utilization (last phase) was found to be constrained completely by the oxygen mass transfer coefficient.

The sensitivity to the various parameters was studied, and it was found that the dynamic model was most sensitive to $k _ { \mathrm { L } } a$ whereas it was less sensitive to other parameters. The importance of the objective function was examined, and it was found that an instantaneous objective function was more representative of the experimental results than an end-point objective function. Another advantage of dFBA is that can incorporate kinetic expressions for reactions that are well-studied. This approach could also be used to identify regulatory phenomena and obtain insight into the functioning of the metabolic pathways. Changes in the regulatory structure that optimize the dynamics of a particular metabolic process could be obtained as a solution to a modified DFBA problem.

In conclusion, we have presented analysis tools for the quantitative study of the dynamic reprogramming of metabolic networks. These tools, along with experimental technologies such as microarrays, GeneChips, and proteomics, will help further understanding of the dynamic behavior of metabolic networks. Additionally, the DFBA approach can be used to provide strategies for the design of a network with a desired objective for metabolic engineering. Finally, the DFBA approach is an extension to classical FBA and has demonstrated great potential; however, further analysis is needed to improve the predictive capabilities in the biological sciences.

Financial support for this work was provided by the National Science Foundation (BES-9896061 and BES-0120241) and the US Department of Energy, Office of Biological and Environmental Research (Microbial Cell Project).

# REFERENCES

Bonarius, H. P. J., G. Schmid, and J. Tramper. 1997. Flux analysis of undetermined metabolic networks: the quest for the missing constraints. Trends Biochem. Sci. 15:308 –314.   
Cohen, B. A., R. D. Mitra, and J. D. Hughes. 2000. A computational analysis of whole-genome expression data reveals domains of gene expression. Nat. Genet. 26:183–186.   
Cronan, J. E., Jr., and D. Laporte. 1996. Tricarboxylic acid cycle and glyoxalate bypass. In Escherichia coli and Salmonella: Cellular and Molecular Biology. F. C. Neidhardt and H. E. Umbarger, Eds. Second ed. ASM Press, Washington, D. C. 189 –198.   
Cuthrell, J. E., and L. T. Biegler. 1987. On the optimization of differential algebraic process systems. AIChE J. 33:1257–1270.   
De Saizieu, A., U. Certa, J. Warrington, C. Gray, W. Keck, and J. Mous. 1998. Bacterial transcript imaging by hybridization of total RNA to oligonucleotide arrays. Nature Biotechnol. 16:45– 48.   
Dhurjati, P., D. Ramkrishna, M. C. Flickinger, and G. T. Tsao. 1985. A cybernetic view of microbial growth: modeling of cells as optimal strategists. Biotech. Bioeng. 27:1–9.   
Edwards, J. S., R. Ramakrishna, C. H. Schilling, and B. O. Palsson. 1999. Metabolic flux balance analysis. In Metabolic Engineering. S. Y. Lee and E. T. Papoutsakis, Eds. Marcel Dekker, New York. 13–57.   
Edwards, J. S., R. U. Ibarra, and B. O. Palsson. 2001. In silico predictions of Escherichia coli metabolic capabilities are consistent with experimental data. Nature Biotechnol. 19:125–130.   
Fell, D. 1996. Understanding the Control of Metabolism. Portland Press, London.   
Giuseppin, M. L. F., and N. A. W. van Riel. 2000. Metabolic modeling of Saccharomyces cerevisiae using the optimal control of homeostasis: a cybernetic model definition. Metabolic Eng. 2:14 –33.   
Guardia, M. J., and E. G. Calvo. 2001. Modeling of Escherichia coli growth and acetate formation under different operational conditions. Enzyme. Microb. Technol. 29:449 – 455.   
Ideker, T., V. Thorsson, J. A. Ranish, R. Christmas, J. Buhler, J. K. Eng, R. Bumgarner, D. R. Goodlett, R. Aebersold, and L. Hood. 2001. Integrated genomic and proteomic analyses of a systematically perturbed network. Science. 292:929 –934.   
Kirkpatrick, C., L. M. Maurer, N. E. Oyelakin, Y. N. Yoncheva, R. Maurer, and J. L. Slonzewski. 2001. Acetate and formate stress: opposite responses in the proteome of Escherichia coli. J. Bacteriol. 183: 6466 – 6477.   
Kompala, D. S. 1984. Bacterial growth on multiple substrates. Experimental verification of cybernetic models. Ph.D. thesis. Purdue University, West Lafayette, IN.   
Lendenmann, U., and T. Egli. 1998. Kinetic models for the growth of Escherichia coli with mixtures of sugars under carbon-limited conditions. Biotech. Bioeng. 59:99 –107.   
Narang, A., A. Konopka, and D. Ramkrishna. 1997. New patterns of mixed-substrate utilization during batch growth of Escherichia coli K12. Biotech. Bioeng. 55:747–757.   
Oh, M. K., and J. C. Liao. 2000. Gene expression profiling by DNA microarrays and metabolic fluxes in Escherichia coli. Biotechnol. Prog. 16:278 –286.   
Pramanik, J., and J. D. Keasling. 1997. Stoichiometric model of Escherichia coli metabolism: incorporation of growth-rate dependent biomass composition and mechanistic energy requirements. Biotech. Bioeng. 56:398 – 421.   
Ramakrishna, R., A. Konopka, and D. Ramkrishna. 1996. Cybernetic modeling of growth in mixed, substitutable substrate environments: preferential and simultaneous utlization. Biotech. Bioeng. 52:141–151.   
Savageau, M. A., E. O. Voit, and D. H. Irvine. 1987a. Biochemical systems theory and metabolic control theory: 1. Fundamental similarities and differences. Math. Biosci. 86:127–145.   
Savageau, M. A., E. O. Voit, and D. H. Irvine. 1987b. Biochemical systems theory and metabolic control theory: 2. The role of summation and connectivity relationships. Math. Biosci. 86:147–169.   
Schilling, C. H., D. Letscher, and B. O. Palsson. 2000a. Theory for the systematic definition of metabolic pathways and their use in interpreting metabolic function from a pathway oriented perspective. J. Theor. Biol. 203:229 –248.   
Schilling, C. H., J. S. Edwards, D. Letscher, and B. O. Palsson. 2000b. Combining pathway analysis with flux balance analysis for the comprehensive study of metabolic systems. Biotech. Bioeng. 71:286 –306.   
Selinger, D. W., K. J. Cheung, R. Mei, E. M. Johansson, C. S. Richmod, F. R. Blattner, D. J. Lockhart, and G. M. Church. 2000. RNA expression analysis using a 30 base pair resolution Escherichia coli genome array. Nature Biotechnol. 18:1262–1268.   
Stephanpoulos, G. 1999. Metabolic fluxes and metabolic engineering. Metab. Eng. 1:1–11.   
Tao, H., C. Bausch, C. Richmond, F. R. Blattner, and T. Conway. 1999. Functional genomics: expression analysis of Escherichia coli growing on minimal and rich media. J. Bacteriol. 181:6425– 6440.   
Tao, H., R. Gonzalez, A. Martinez, M. Rodriguez, L. O. Ingram, J. F. Preston, and K. T. Shanmugam. 2001. Engineering a homo-ethanol pathway in Escherichia coli: increased glycolytic flux and levels of expression of glycolytic genes during xylose fermentation. J. Bacteriol. 183:2979 –2988.   
Tavazoie, S., J. D. Hughes, M. J. Campbell, R. J. Cho, and G. M. Church. 1999. Systematic determination of genetic network architecture. Nat. Genet. 22:281–285.   
Torres, N. V. 1994. Application of the transition time of metabolic systems as a criterion for optimization of metabolic processes. Biotech. Bioeng. 44:291–296.   
van Riel, N. A. W., M. L. F. Giuseppin, and C. T. Verrips. 2000. Dynamic optimal control of homeostasis: an integrative system approach for modeling of the central nitrogen metabolism in Saccharomyces cerevisiae. Metabolic Eng. 2:14 –33.   
Varma, A., and B. O. Palsson. 1994a. Metabolic flux balancing— basic concepts, scientific and practical use. Bio-Technol. 12:994 –998.   
Varma, A., and B. O. Palsson. 1994b. Stoichiometric flux balance models quantitatively predict growth and metabolic by-product secretion in wild-type Escherichia coli W3110. Appl. Environ. Microbiol. 60: 3724 –3731.   
Varma, A., B. W. Boesch, and B. O. Palsson. 1993. Stoichiometric interpretation of Escherichia coli glucose catabolism under various oxygenation rates. Appl. Environ. Microbiol. 59:2465–2473.   
Varner, J. 2000. Large-scale prediction of phenotype: concept. Biotech. Bioeng. 69:664 – 678.   
Varner, J., and D. Ramkrishna. 1999. Metabolic engineering from a cybernetic perspective I. Theoretical preliminaries. Biotechnol. Prog. 15: 407– 425.   
Wei, Y., J. M. Lee, C. Richmond, F. R. Blattner, J. A. Rafalski, and R. A. LaRossa. 2001. High-density microarray-mediated gene expression profiling of Escherichia coli. J. Bacteriol. 183:545–556.   
Wong, P., S. Gladney, and J. D. Keasling. 1997. Mathematical Model of the lac Operon: inducer exclusion, catabolite repression, and diauxic growth on glucose and lactose. Biotechnol. Prog. 13:132–143.