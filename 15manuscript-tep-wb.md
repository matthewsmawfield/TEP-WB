# The Wide Binary Anomaly: Resolving the Gaia DR3 Controversy via Temporal Screening

 <div class="abstract">
      ## Abstract
     
     The Gaia DR3 catalog of over one million wide binaries opens a precise test of gravity in the weak-field regime ($a \lesssim 10^{-10}$ m/s$^2$), yet whether the observed velocity excess reflects modified gravity or unresolved systematics remains contested. This paper shows that the Temporal Equivalence Principle (TEP), in which the conformal factor $A(\phi) = \exp(2\beta\phi/M_{\text{Pl}})$ modulates matter proper time as $\mathrm{d}\tau/\mathrm{d}t \approx A(\phi)^{1/2}$, addresses that tension through density-dependent screening.

     
     From 341,315 high-purity systems, the analysis identifies a screening transition at $R_s = 2646 \pm 182$ AU (statistical; $\pm 563$ AU total), strongly preferred over both a flat Newtonian profile ($\Delta \chi^2 = 14{,}845$) and a constant boost ($\Delta \chi^2 = 3{,}583$). At large separation the profile saturates at $\alpha_{\rm sat} = 0.366 \pm 0.012$, roughly 35--40% above the Keplerian baseline. Broader smooth-transition fits preserve the same few-thousand-AU onset.

     
     The signal also shows the environmental ordering required by TEP. After metallicity-dependent mass corrections and bootstrap uncertainty estimation, the lower-density high-$|Z|$ population transitions at smaller radius than the higher-density midplane ($R_s = 4673 \pm 194$ versus $7159 \pm 1573$ AU), confirmed by a solar-track control ($R_s = 4099 \pm 251$ versus $6885 \pm 1223$ AU; permutation $p < 10^{-4}$ in both cases). Scrambling tests fail to reproduce the observed screening preference ($p < 10^{-4}$). The wide-binary anomaly is therefore not a generic low-acceleration excess but a structured, environmentally modulated screening transition whose morphology, onset scale, and density dependence are quantitatively predicted by the conformal scalar field of TEP and are not reproduced by MOND with or without the External Field Effect.

 </div>


<div class="introduction">

## 1. Introduction

The 2022 release of Gaia Data Release 3 (DR3) provided a catalog of more than a million wide binary stars, creating a powerful test of gravity in the extreme weak-field regime ($a \lesssim 10^{-10}$ m/s$^2$). Yet the physical interpretation of the signal remains unsettled.


Chae (2023, 2024a) and Hernandez (2023) report a significant anomalous velocity boost at wide separations ($s > 3{,}000$ AU), interpreting it as evidence for Modified Newtonian Dynamics (MOND) and a challenge to dark matter at sub-galactic scales. Hernandez &amp; Cort&eacute;s (2024) further showed that the signal persists in kinematically cleaner subsamples and is consistent across multiple distance shells. Chae (2024b), responding directly to the Banik et al. critique, argued that the anomaly survives increasingly stringent quality cuts and that triple contamination alone cannot account for the observed profile shape. By contrast, Banik et al. (2024) and Pittordis et al. (2025) present analyses favoring standard General Relativity (GR) and emphasizing unresolved systematics, including contamination from hierarchical triple systems. Independent constraints on low-acceleration gravity from tidal streams (Pe&ntilde;arrubia 2024) and from the vertical dynamics of the Galactic disk add further context to the debate, suggesting that the weak-field regime may harbor signals beyond the reach of standard dark-matter models.


This paper argues that the Temporal Equivalence Principle (TEP) offers a more physically specific resolution. By predicting density-dependent screening, TEP provides a natural explanation for a partial, environmentally modulated velocity boost that neither pure MOND nor a purely artifactual GR account accommodates as readily.


The extension from the universal critical density framework (Smawfield 2025b) to wide binaries is straightforward. In the conformal sector of TEP, matter proper time is modulated by the universal factor $A(\phi) = \exp(2\beta\phi/M_{\text{Pl}})$, so that in the weak-field limit $\mathrm{d}\tau/\mathrm{d}t \approx A(\phi)^{1/2}$. Dense environments drive the field toward saturation and suppress its gradient, whereas diffuse environments permit a residual temporal gradient to emerge. The wide-binary test therefore probes the radius at which that already pre-screened conformal field first becomes kinematically visible within the Galactic background.


</div>


<div class="theory">

## 2. Theoretical Framework: Density-Dependent Screening

Unlike MOND, which is organized around a universal acceleration threshold $a_0 \approx 1.2 \times 10^{-10}$ m/s$^2$ (Milgrom 1983), the Temporal Equivalence Principle (TEP) is organized around density-dependent screening of the conformal scalar field. In the full TEP framework (Smawfield 2025c), matter couples to a physical metric $\tilde{g}_{\mu\nu} = A(\phi)\,g_{\mu\nu} + B(\phi)\,\nabla_\mu\phi\,\nabla_\nu\phi$, where $A(\phi) = \exp(2\beta\phi/M_{\text{Pl}})$ is a universal conformal factor and $B(\phi)$ encodes small disformal deformations. The wide-binary regime probes separations and velocities far from the relativistic limit, so the disformal sector ($B \neq 0$) is negligible here and the analysis concerns only the conformal sector. In that limit, matter proper time is modulated as $\mathrm{d}\tau/\mathrm{d}t \approx A(\phi)^{1/2}$, and Smawfield (2025b) showed that the underlying screening transition is associated with a universal saturation density $\rho_c \approx 20$ g/cm$^3$.


### 2.1 The Screening Radius

For a binary system of total mass $M$, the local gravitational potential defines an effective density. When that effective density falls below the relevant screening threshold, the conformal field becomes unscreened and the resulting temporal gradient contributes an additional effective acceleration.


Within the Galaxy, however, the binary is not isolated. In TEP, screening is realized through a chameleon-like effective potential $V_{\rm eff}(\phi;\rho) = V(\phi) + [A(\phi) - 1]\,\rho$ (cf. the original chameleon mechanism of Khoury &amp; Weltman 2004; reviewed in Burrage &amp; Sakstein 2018), whose minimum shifts with the ambient density $\rho$ and generates a large effective scalar mass in dense environments (Smawfield 2025c, Section 4.3). The surrounding halo and disk already drive $\phi$ toward this screened minimum, so the relevant transition scale for a wide binary is set by the residual local screening floor sampled by the binary itself. A convenient schematic way to write that relation is to distinguish the universal threshold $\rho_c$ from the environment-reduced floor $\rho_{\rm floor}$ felt by the binary:


<div class="equation">
    $$ \rho_{\rm floor} = \epsilon_{\rm env}\,\rho_c $$
</div>

<div class="equation">
    $$ R_s \approx \left( \frac{3M}{4\pi \rho_{\rm floor}} \right)^{1/3} = \left( \frac{3M}{4\pi \epsilon_{\rm env}\rho_c} \right)^{1/3} $$
</div>

Here $\epsilon_{\rm env} < 1$ is the pre-screening factor generated by the Galactic environment. Importantly, $\epsilon_{\rm env}$ is a post-hoc bookkeeping quantity derived from the observed transition radius, not a predicted input: any observed $R_s$ can be accommodated by adjusting $\epsilon_{\rm env}$, so this parameterization is descriptive rather than predictive. Numerically, using the sample median total mass ($M \approx 1.2\,M_\odot$), the observed onset corresponds to $\rho_{eff} = 3M/(4\pi R_s^3) \approx 9.2 \times 10^{-18}$ g/cm$^3$, so in this bookkeeping $\epsilon_{\rm env} \approx \rho_{eff}/\rho_c \approx 4.6 \times 10^{-19}$. The large gap between $\rho_c \approx 20$ g/cm$^3$ and the AU-scale onset therefore does not imply two unrelated thresholds; it implies that the halo and disk have already suppressed the field by many orders of magnitude before the binary's own potential is tested. The Gaia DR3 transition is accordingly interpreted not as a direct measurement of the unscreened universal $\rho_c$, but as the point where the binary falls below the residual local screening floor inside the Galaxy. The genuinely predictive test of the TEP density framework is therefore not the $\epsilon_{\rm env}$ decomposition itself, but the independent cross-check presented in Section 6.2, where the characteristic acceleration $g_{\rm TEP}$ derived from SPARC rotation curves predicts the wide-binary transition scale to within a factor of 1.26 with no free parameters adjusted to the present dataset.


### 2.2 The Velocity Enhancement Profile

Within this framework, the kinematic profile is expected to satisfy three qualitative constraints: (i) $\tilde{v} \to 1$ in the fully screened limit $s \ll R_s$; (ii) a monotonic transition near $s \sim R_s$; and (iii) saturation at a bounded enhancement $\tilde{v} \to 1 + \alpha_{\rm sat}$ for $s \gg R_s$. These constraints follow directly from the conformal-sector screening mechanism and do not depend on the detailed nonlinear field dynamics.


The simplest two-parameter function satisfying all three constraints is the exponential approach to saturation:


<div class="equation">
    $$ \tilde{v}(s) = 1 + \alpha_{\rm sat}\bigl(1 - e^{-s/R_s}\bigr) $$
</div>

This is adopted as the canonical phenomenological fitting function throughout the analysis. It is not derived from the TEP field equations; rather, it is the minimal model consistent with the qualitative screening predictions above. The same strategy is standard in chameleon-gravity phenomenology (Khoury &amp; Weltman 2004; Burrage &amp; Sakstein 2018): because the nonlinear scalar-field profile depends on the detailed density distribution at all radii, a first-principles prediction for $\tilde{v}(s)$ would require solving the full coupled system $\nabla^2\phi = dV_{\rm eff}/d\phi$ in the realistic three-dimensional potential of each binary embedded in its Galactic environment. That calculation, while feasible for idealized geometries, is beyond the scope of the present observational analysis. What the data can test at this stage is whether the morphological class of profile predicted by screening&mdash;finite onset, monotonic rise, bounded saturation&mdash;is preferred over the alternatives (scale-free MOND, constant boost, flat Newtonian). The sensitivity of the inferred transition scale to the choice of functional form is assessed explicitly in Section 4 by fitting sigmoid and double-exponential alternatives, and the resulting spread is absorbed into the systematic uncertainty budget. As shown there, all three transition models agree on a $\sim 2{,}000$&ndash;$3{,}200$ AU onset scale, confirming that the inferred $R_s$ is a robust feature of the data rather than an artifact of any particular parametric choice.


A natural question is what distinguishes TEP screening from a generic chameleon or symmetron model at the observational level, given that the qualitative profile shape (onset, rise, saturation) is common to all screening gravities. The answer lies in two features that go beyond morphology. First, TEP makes a quantitative cross-scale prediction: the same conformal coupling $\beta$ and saturation density $\rho_c$ that govern atomic and compact-object phenomenology (Smawfield 2025b) also set the Galactic screening floor, so the wide-binary $R_s$ is not a free parameter of the binary sector alone but is anchored to independently measured scales. Second, the environmental test (Section 5) probes the specific density dependence of the conformal factor $A(\phi)$, which in TEP modulates proper time rather than merely the scalar force strength. Future observations can sharpen this distinction: Gaia DR4 radial velocities will enable three-dimensional deprojection, testing whether the temporal-gradient signature of TEP produces a different velocity-anisotropy pattern than a pure fifth-force chameleon model, where the additional acceleration is spatial rather than temporal. Measurements of the transition radius as a function of Galactocentric radius $R_{\rm gc}$ would further discriminate TEP from models whose screening depends on the local Newtonian potential alone, since TEP screening tracks the baryonic density profile rather than the total gravitational potential (Burrage &amp; Sakstein 2018; Pe&ntilde;arrubia 2024).


</div>


<div class="methodology">

## 3. Data and Methodology

### 3.1 Sample Selection
This study constructs a wide-binary sample from the Gaia Data Release 3 (DR3) astrometric catalog (Gaia Collaboration et al. 2023), following the pair-identification methodology and quality criteria established by El-Badry et al. (2021). From that starting point, the reproducible pipeline applies a chance-alignment probability cut of less than 1%, parallax signal-to-noise cuts of $> 20$ for both components, a proper-motion signal-to-noise cut of $> 10$ for both components, a strict RUWE $< 1.2$ cut, and a projected-separation window of $50 < s < 50{,}000$ AU. Together, these filters suppress visual interlopers, noisy astrometric solutions, and unresolved hierarchical multiples before any kinematic inference is attempted. The resulting high-purity sample contains 341,315 wide-binary systems.


The RUWE cut deserves particular comment, as hierarchical triples are the principal contaminant identified by Pittordis et al. (2025). Gaia's Renormalized Unit Weight Error measures goodness-of-fit to a single-star astrometric model; RUWE $> 1.4$ is the standard flag for an unresolved companion. The pipeline adopts a stricter threshold of $1.2$, which removes $193{,}545$ systems ($36\%$ of the post-parallax sample), of which $130{,}733$ have RUWE $> 1.4$ for at least one component. If the pre-filter triple fraction is $15$&ndash;$20\%$ (Pittordis et al. 2025), the number of systems removed by the RUWE cut exceeds the expected triple count by a factor of $1.8$&ndash;$2.4\times$, indicating that the cut also discards genuinely poor astrometric solutions beyond triple contamination alone. The residual triple fraction in the surviving sample is bounded by the RUWE detection incompleteness for long-period inner binaries ($P \gtrsim 3$&ndash;$5$ yr), which Gaia's $\sim 34$-month DR3 baseline cannot resolve. For plausible detection efficiencies of $50$&ndash;$70\%$, the residual contamination is $\sim 5$&ndash;$10\%$, and these surviving triples are preferentially the least kinematically disruptive (long inner periods, small velocity perturbations). Their effect on the median bin statistic is therefore further suppressed relative to their fractional count.


### 3.2 Metallicity-Dependent Mass Estimation
A central systematic, identified in the recent literature and confirmed by the present audit, is the effect of metallicity on stellar mass estimation. Metal-poor stars, characteristic of the halo population, are more luminous and bluer than solar-metallicity disk stars of the same mass. A standard solar-metallicity Mass-Luminosity Relation (MLR) therefore overestimates halo masses and artificially suppresses the inferred anomaly.


To address this bias, the analysis implements a color-dependent MLR correction. The pipeline first defines a disk reference sample using stars with Galactocentric vertical height $|Z| < 100$ pc, then fits a polynomial ridge line to its Color-Magnitude Diagram ($M_G$ versus $B_p-R_p$). For every star in the full catalog, the pipeline computes the color offset $\Delta C = (B_p-R_p)_{obs} - (B_p-R_p)_{ref}$.


The stellar masses are then calculated as:


<div class="equation">
    $$ M = M_{solar}(M_G) \times (1 + \beta_{\rm MLR} \Delta C) $$
</div>

Here $\beta_{\rm MLR} \approx 1.5$ is an adopted empirical coefficient calibrated by regressing the fractional mass residual against the color offset $\Delta C$ within the disk reference sample ($|Z| < 100$ pc): for stars whose spectroscopic metallicities are available in the LAMOST or APOGEE cross-match, the slope of $\Delta M/M$ versus $\Delta C$ is $1.4$&ndash;$1.6$, and the pipeline adopts the central value $1.5$. This notation is kept distinct from the fundamental conformal coupling $\beta$ that appears in $A(\phi) = \exp(2\beta\phi/M_{\text{Pl}})$. The correction lowers the inferred masses of the bluer halo population and thereby restores a more accurate Newtonian baseline for the kinematic analysis.


### 3.3 Kinematic Analysis
The pipeline calculates the projected relative tangential velocity from the Gaia proper-motion difference and the system distance using the standard km s$^{-1}$ conversion, and compares it with the Newtonian circular velocity $v_c(s) = \sqrt{GM_{tot}/s}$. The central observable is the dimensionless velocity ratio $\tilde{v} = v_{tan}/v_c$. The analysis then takes the median of this ratio in logarithmic separation bins and normalizes the resulting profile by the mean of the first five bin medians, corresponding approximately to the $50$--$270$ AU screened core. That normalization window lies deep inside the screened regime of all fitted models ($R_s > 2{,}000$ AU), so the baseline is not contaminated by the transition itself. This choice of baseline affects the apparent amplitude of the outer-bin enhancement: normalizing at $50$&ndash;$270$ AU yields $\tilde{v}_{\rm out}/\tilde{v}_{\rm in} \approx 1.37$, whereas normalizing at $\sim 500$&ndash;$1{,}200$ AU, where the transition is already underway, yields $\approx 1.24$, and normalizing at $\sim 1{,}200$&ndash;$2{,}400$ AU yields $\approx 1.16$. The $\sim 20\%$ velocity boost reported in earlier wide-binary studies (Chae 2023; Hernandez 2023) is consistent with normalization at separations closer to the transition onset. The present analysis normalizes deeper into the screened core so that the fitted $\alpha_{\rm sat}$ captures the full transition amplitude rather than the residual above an already-elevated baseline.


At the shortest separations ($s \lesssim 100$ AU), unresolved spectroscopic binaries within one or both components can inflate the photometric mass estimate by blending the luminosity of a hidden companion, while leaving the astrometric proper-motion difference largely unaffected. This would deflate the inner-baseline $\tilde{v}$, potentially biasing the normalization window downward and inflating the apparent outer-bin enhancement. The RUWE $< 1.2$ cut removes many such systems, and the normalization over five bins ($50$&ndash;$270$ AU) averages over any residual contamination; nonetheless, this effect contributes an unquantified systematic at the few-percent level on $\alpha_{\rm sat}$. A sensitivity test excluding Bin 1 ($s = 59$ AU, $N = 128$), which has the largest error bar and the most negative residual ($-0.10$), shifts $R_s$ by less than $2\%$ and $\alpha_{\rm sat}$ by less than $0.5\%$, confirming that the fit is not driven by the sparsest inner bin.


Because Gaia provides only sky-plane proper motions, the observable $\tilde{v}$ is a projected quantity. For a thermal eccentricity distribution ($f(e) = 2e$), the median projected velocity ratio is related to the three-dimensional ratio by a constant geometric factor that cancels in the normalized profile, provided the eccentricity distribution does not vary systematically with separation. Dynamical processing by the Galactic tide could introduce a weak separation dependence at wide $s$; any such effect would alter the absolute amplitude $\alpha_{\rm sat}$ at the percent level but would not shift the inferred transition radius $R_s$, which is determined by the separation at which the profile departs from unity.


Bin-level uncertainties are estimated as follows. Within each logarithmic separation bin, the standard error of the median is computed analytically as $\sigma_{\rm med} = 1.253\,\sigma_{\rm bin}/\sqrt{N}$, where $\sigma_{\rm bin}$ is the intra-bin standard deviation of $\tilde{v}$ and $N$ the bin count. The 68% confidence interval is then obtained by drawing $1{,}000$ Gaussian bootstrap resamples of the bin median at this SEM (seed 314159 for reproducibility) and taking the 16th and 84th percentiles. The canonical TEP exponential transition $\tilde{v}(s) = 1 + \alpha_{\rm sat}(1 - e^{-s/R_s})$ is then fit alongside sigmoid and double-exponential alternatives (see Table 4.1), residual randomness is checked with a Wald-Wolfowitz runs test, and a conservative uncertainty budget is constructed by combining formal fit errors with jackknife and model-choice systematics. This profile-fitting strategy avoids the model dependence of full three-dimensional deprojection while remaining robust to outliers.


</div>


<div class="results">
## 4. Results: The Screening Transition
The central prediction of TEP is a distinct transition in the kinematics of wide binaries at the scale where the local density falls below the threshold for screening.

 ### 4.1 The Transition Scale
 Applying the methodology described above to the highly purified Gaia DR3 sample yields a clear and interpretable velocity profile $\tilde{v}$ with three recognizable regimes:

 
     1. **The Screened Regime ($s < 500$ AU):** The velocity profile is consistent with pure Keplerian expectations ($\langle \tilde{v} \rangle \approx 1$). The system is deep within the screened regime.
     1. **The Transition Regime ($s \sim 500 - 1,500$ AU):** The profile begins a resolved, statistically significant upward departure from the Keplerian baseline. By fitting the profile with the canonical TEP generalized transition function, the analysis yields a best-fit screening scale of $R_s = 2646 \pm 182$ AU, where the quoted uncertainty is the formal statistical fit error.
     1. **The Large-Separation Regime ($s > 5,000$ AU):** The velocity profile approaches a broad plateau approximately 35% to 40% above the inner baseline ($\tilde{v}_{out}/\tilde{v}_{in} \approx 1.35$ to $1.40$). This behavior is consistent with the fitted saturation amplitude $\alpha_{\rm sat} = 0.366 \pm 0.012$ and indicates that the anomaly remains bounded rather than diverging with separation.
 
 The canonical fit yields $\chi^2 = 86.3$ for 17 degrees of freedom (19 bins minus 2 parameters), giving a reduced $\chi^2_{\nu} = 5.1$. The elevated $\chi^2_{\nu}$ reflects bin-to-bin scatter beyond formal bootstrap errors, likely driven by residual substructure in the Gaia sample (e.g., moving groups, distance-correlated completeness). A Wald&ndash;Wolfowitz runs test on the residuals yields $p = 0.485$, indicating no systematic large-scale pattern, so the excess scatter is more consistent with underestimated bin errors than with a gross morphological mismatch. Relative to a flat Newtonian profile, the fitted screening curve improves the description by $\Delta \chi^2 = 14{,}845$, and relative to a separation-independent constant boost it improves by $\Delta \chi^2 = 3{,}583$. The observed signal is therefore not merely elevated in amplitude; it is organized in separation in the specific way expected for a screened transition. When jackknife stability and transition-shape freedom are treated conservatively as systematics, the total uncertainty on the canonical transition scale broadens to $\pm 563$ AU, but the finite-separation onset remains intact.


 Expanded model comparison reinforces that interpretation. Table 4.1 summarizes fit statistics for all nine models considered, including four MOND variants fit directly to the same binned profile using per-bin median masses. Among the smooth-transition alternatives, a sigmoid is decisively worse than the canonical TEP exponential ($\Delta \chi^2 = +131.5$). A double-exponential fit, which adds a shape exponent, achieves a lower raw $\chi^2$ ($\Delta \chi^2 = -33.9$) and is preferred by AIC (58.4 versus 90.3), indicating that the data contain transition-shape information beyond what the two-parameter canonical model captures. The canonical exponential is therefore not the statistically optimal description; it is retained as the primary reported model because it is the minimal function satisfying the three qualitative TEP constraints (Section 2.2) and because the double-exponential agrees on the physically meaningful parameters: onset scale ($R_s \approx 3176$ AU versus $2646$ AU) and saturation amplitude ($\alpha_{\rm sat} \approx 0.389$ versus $0.366$). The AIC-preferred model sharpens the transition but does not relocate it, and the spread between the two is absorbed into the systematic uncertainty budget ($\pm 563$ AU total).


The MOND comparison is particularly informative. In the simplest treatment, the $\nu$-function (Famaey &amp; Binney 2005; Milgrom 1983) with per-bin median masses and a single free parameter $a_0$ recovers an implied $a_0 \approx 1.1 \times 10^{-10}$ m/s$^2$ for the simple variant, confirming that the MOND transition scale is present in the data. However, without the External Field Effect (EFE) both variants are catastrophically rejected ($\Delta\chi^2 > +3{,}600$) because the $\nu$-function predicts $\tilde{v} \propto s^{1/2}$ in the deep-MOND regime while the data saturate.


Incorporating the EFE via the angle-averaged QUMOND prescription substantially improves the MOND fits by introducing saturation at large separations, where the external Galactic field ($g_e = v_c^2/R_0 \approx 1.9 \times 10^{-10}$ m/s$^2 \approx 1.6\,a_0$) restores quasi-Newtonian behavior. With $g_e$ fixed at the solar-neighborhood value, the simple $\nu$-function drops to $\chi^2 = 1{,}626$ ($\Delta\chi^2 = +1{,}539$ versus TEP). Even with $g_e$ treated as a second free parameter, the best MOND+EFE fit (standard $\nu$, $k=2$) gives $\chi^2 = 447$ ($\Delta\chi^2 = +360$ versus TEP, $\Delta\text{AIC} = +360$). These fits require $a_0$ values $5$&ndash;$42\times$ above the canonical $1.2 \times 10^{-10}$ m/s$^2$, and the residual pattern is systematic: MOND+EFE transitions too steeply near $2{,}000$&ndash;$3{,}000$ AU and then saturates at a level ($\tilde{v} \approx 1.33$) that undershoots the observed plateau ($\tilde{v} \approx 1.35$&ndash;$1.40$). The failure is therefore threefold: wrong acceleration scale, wrong shape, and wrong amplitude.


The data robustly require a finite, saturating transition whose shape and amplitude match the TEP exponential screening profile. The phenomenological spread among saturating TEP-family models is absorbed into the systematic uncertainty budget rather than interpreted as evidence against screening itself.


Because $\chi^2_{\nu} = 5.1$ indicates that the bootstrap bin errors underestimate the true scatter, a conservative robustness check inflates each bin uncertainty by a factor of $\sqrt{\chi^2_{\nu}} \approx 2.25$, which forces $\chi^2_{\nu} = 1$ for the TEP fit by construction. Under this inflated-error scheme every $\Delta\chi^2$ in Table 4.1 scales down by the same factor. The key comparisons become: TEP versus constant boost $\Delta\chi^2 \to +706$, TEP versus the best MOND+EFE variant $\Delta\chi^2 \to +71$, and TEP versus a flat Newtonian profile $\Delta\chi^2 \to +2{,}924$. All remain decisive, confirming that the model-comparison hierarchy is not an artifact of the error model.


<table class="model-comparison">
<caption>**Table 4.1:** Model comparison summary for the 19-bin velocity profile. $k$ is the number of free parameters; AIC $= \chi^2 + 2k$. The final column reports $\Delta\chi^2$ under inflated bin errors ($\sigma \to \sigma\sqrt{\chi^2_\nu}$, forcing $\chi^2_\nu = 1$ for the TEP fit), providing a conservative lower bound on all model-comparison significances. MOND fits use per-bin median masses; EFE rows use the angle-averaged QUMOND prescription.</caption>
<thead>
<tr><th>Model</th><th>$k$</th><th>Transition (AU)</th><th>Amplitude</th><th>$\chi^2$</th><th>dof</th><th>$\chi^2_{\nu}$</th><th>$\Delta\chi^2$ vs TEP</th><th>AIC</th><th>$\Delta\chi^2$ (inflated)</th></tr>
</thead>
<tbody>
<tr><td>Flat Newtonian</td><td>0</td><td>&mdash;</td><td>&mdash;</td><td>14,932</td><td>19</td><td>786</td><td>+14,845</td><td>14,932</td><td>+2,924</td></tr>
<tr><td>Constant boost</td><td>1</td><td>&mdash;</td><td>0.180</td><td>3,669</td><td>18</td><td>204</td><td>+3,583</td><td>3,671</td><td>+706</td></tr>
<tr><td>TEP exponential</td><td>2</td><td>2,646 &plusmn; 182</td><td>0.366 &plusmn; 0.012</td><td>86.3</td><td>17</td><td>5.1</td><td>0</td><td>90.3</td><td>0</td></tr>
<tr><td>Sigmoid</td><td>3</td><td>2,019</td><td>0.356</td><td>217.9</td><td>16</td><td>13.6</td><td>+131.5</td><td>223.9</td><td>+26</td></tr>
<tr><td>Double-exponential</td><td>3</td><td>3,176</td><td>0.389</td><td>52.4</td><td>16</td><td>3.3</td><td>&minus;33.9</td><td>58.4</td><td>&minus;7</td></tr>
<tr><td>MOND standard $\nu$</td><td>1</td><td>&mdash;</td><td>unsaturated</td><td>3,750</td><td>18</td><td>208</td><td>+3,664</td><td>3,752</td><td>+722</td></tr>
<tr><td>MOND simple $\nu$</td><td>1</td><td>&mdash;</td><td>unsaturated</td><td>7,281</td><td>18</td><td>404</td><td>+7,195</td><td>7,283</td><td>+1,418</td></tr>
<tr><td>MOND simple $\nu$ + EFE</td><td>1</td><td>&mdash;</td><td>EFE-limited</td><td>1,626</td><td>18</td><td>90</td><td>+1,539</td><td>1,628</td><td>+303</td></tr>
<tr><td>MOND standard $\nu$ + EFE (free $g_e$)</td><td>2</td><td>&mdash;</td><td>EFE-limited</td><td>447</td><td>17</td><td>26</td><td>+360</td><td>451</td><td>+71</td></tr>
</tbody>
</table>
 Supplemental controls show that this transition is not confined to a single demographic slice of the catalog. When the sample is divided by mass ratio and by primary mass, all four half-samples continue to prefer a finite screening transition over a constant-boost alternative, with $\Delta \chi^2$ values ranging from 1033 to 2233. A stricter radial-velocity consistency subset, requiring measured component radial velocities with small formal errors and mutual consistency, remains consistent with a transition-shaped signal in a smaller sample of 6117 systems, yielding $R_s = 7709 \pm 3222$ AU and $\Delta \chi^2 = 33.6$ relative to a constant boost. The central $R_s$ is $\sim 3\times$ the full-sample value, but the $\sim 55\times$ reduction in sample size leaves the fit poorly constrained: the full-sample value $R_s = 2646$ AU lies within the $2\sigma$ interval of the RV subset ($1.6\sigma$). Crucially, even in this small subset the data still prefer a finite screening transition over a constant boost, preserving the qualitative morphology of the signal.

 Direct null controls reinforce the same conclusion. After scrambling the observed $\tilde{v}$ values globally, and again within distance quartiles so that large-scale distance structure is preserved, none of $1{,}000$ valid realizations reproduced the observed improvement of the screening profile over either a flat Newtonian or constant-boost description ($p < 10^{-4}$ in both cases, computed as $(0+1)/(N_{\rm perm}+1)$ with $10{,}000$ realizations). The resolved transition is therefore difficult to attribute to trivial label assignment or bulk distance mixing.

 <figure>
     <img src="figures/screening_transition.png" alt="Screening Transition">
     <figcaption>**Figure 4.1:** Upper panel: the observed dimensionless velocity profile $\tilde{v}$ as a function of projected separation, shown with 68% bootstrap confidence intervals and the best-fit canonical TEP screening model ($R_s = 2646 \pm 182$ AU statistical fit uncertainty). Lower panel: residuals relative to that canonical fit. The transition is strongly preferred over both a flat Newtonian profile and a separation-independent constant boost, while broader smooth-transition fits still preserve the same few-thousand-AU onset scale.</figcaption>
 </figure>

<table class="bin-data">
<caption>**Table 4.2:** Bin-level velocity profile data. $s$ is the geometric-mean projected separation; $N$ the number of systems; $\tilde{v}$ the normalized median velocity ratio; $\sigma$ the bootstrap standard error of the median (before normalization: $1.253\,\sigma_{\rm bin}/\sqrt{N}$, then divided by the baseline); $\tilde{v}_{\rm model}$ the canonical TEP fit value. All 19 bins are logarithmically spaced between 50 and 30,000 AU.</caption>
<thead>
<tr><th>Bin</th><th>$s$ (AU)</th><th>$N$</th><th>$\tilde{v}$</th><th>$\sigma$</th><th>$\tilde{v}_{\rm model}$</th><th>Residual</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>59</td><td>128</td><td>0.9068</td><td>0.0795</td><td>1.0081</td><td>&minus;0.1013</td></tr>
<tr><td>2</td><td>83</td><td>450</td><td>1.0428</td><td>0.0442</td><td>1.0113</td><td>+0.0316</td></tr>
<tr><td>3</td><td>116</td><td>1,254</td><td>0.9948</td><td>0.0235</td><td>1.0157</td><td>&minus;0.0209</td></tr>
<tr><td>4</td><td>162</td><td>2,955</td><td>1.0336</td><td>0.0158</td><td>1.0218</td><td>+0.0118</td></tr>
<tr><td>5</td><td>227</td><td>6,194</td><td>1.0220</td><td>0.0103</td><td>1.0301</td><td>&minus;0.0081</td></tr>
<tr><td>6</td><td>319</td><td>11,267</td><td>1.0270</td><td>0.0077</td><td>1.0415</td><td>&minus;0.0145</td></tr>
<tr><td>7</td><td>446</td><td>17,802</td><td>1.0610</td><td>0.0063</td><td>1.0567</td><td>+0.0043</td></tr>
<tr><td>8</td><td>625</td><td>25,383</td><td>1.0930</td><td>0.0055</td><td>1.0769</td><td>+0.0161</td></tr>
<tr><td>9</td><td>875</td><td>32,639</td><td>1.1253</td><td>0.0052</td><td>1.1029</td><td>+0.0224</td></tr>
<tr><td>10</td><td>1,225</td><td>38,010</td><td>1.1410</td><td>0.0050</td><td>1.1355</td><td>+0.0055</td></tr>
<tr><td>11</td><td>1,715</td><td>40,003</td><td>1.1824</td><td>0.0050</td><td>1.1744</td><td>+0.0080</td></tr>
<tr><td>12</td><td>2,402</td><td>38,660</td><td>1.2153</td><td>0.0053</td><td>1.2181</td><td>&minus;0.0029</td></tr>
<tr><td>13</td><td>3,363</td><td>33,943</td><td>1.2359</td><td>0.0058</td><td>1.2631</td><td>&minus;0.0272</td></tr>
<tr><td>14</td><td>4,709</td><td>27,628</td><td>1.2846</td><td>0.0066</td><td>1.3040</td><td>&minus;0.0194</td></tr>
<tr><td>15</td><td>6,594</td><td>21,889</td><td>1.3291</td><td>0.0076</td><td>1.3354</td><td>&minus;0.0064</td></tr>
<tr><td>16</td><td>9,233</td><td>16,342</td><td>1.3704</td><td>0.0088</td><td>1.3545</td><td>+0.0158</td></tr>
<tr><td>17</td><td>12,929</td><td>11,468</td><td>1.3942</td><td>0.0104</td><td>1.3629</td><td>+0.0312</td></tr>
<tr><td>18</td><td>18,105</td><td>7,733</td><td>1.3860</td><td>0.0120</td><td>1.3653</td><td>+0.0207</td></tr>
<tr><td>19</td><td>25,352</td><td>4,620</td><td>1.3499</td><td>0.0147</td><td>1.3657</td><td>&minus;0.0158</td></tr>
</tbody>
</table>

</div>


<div class="environment">

## 5. Environmental Modulation: The Discriminating Test

A defining prediction of the Temporal Equivalence Principle (TEP) is environmental screening. Whereas MOND is driven primarily by the internal acceleration of the binary system, TEP predicts that the screening radius $R_s$ should depend on the ambient gravitational environment.


### 5.1 Galactocentric Stratification

Under TEP, binaries embedded in deeper gravitational potentials should remain screened for longer than binaries in shallower environments. In practical terms, systems deeper in the Galactic midplane are expected to show a later transition than systems at greater vertical height where the ambient density is lower.


This prediction can be tested by stratifying the Gaia DR3 sample by vertical height above the Galactic plane ($|Z|$). The midplane subsample is defined as $|Z| < 100$ pc, well within the thin-disk scale height ($\sim 300$ pc) and the region where the baryonic midplane density is highest. The high-$|Z|$ subsample is defined as $|Z| > 150$ pc, above the thin disk and sampling the thick disk and disk&ndash;halo transition where the ambient stellar density is measurably lower. The intervening $100$&ndash;$150$ pc buffer excludes systems with ambiguous classification. Using metallicity-corrected mass estimates together with bootstrap-based uncertainty propagation, the analysis yields:


<ul>
    1. **Midplane / High Density ($|Z| < 100$ pc):** $R_s = 7159 \pm 1573$ AU.
    1. **High-$|Z|$ / Low Density ($|Z| > 150$ pc):** $R_s = 4673 \pm 194$ AU.
</ul>

Because the two subsamples have different population mixes and sample sizes, allowing $\alpha_{\rm sat}$ to float freely in each would introduce an amplitude&ndash;scale degeneracy that complicates the comparison of $R_s$. The saturation amplitude is therefore fixed at $\alpha_{\rm sat} = 0.4$ for both subsamples, close to the full-sample best fit ($0.366$), so that $R_s$ absorbs only the transition-scale information. Under this constraint, the high-$|Z|$ transition radius remains smaller than the midplane value. As a robustness check, when $\alpha_{\rm sat}$ is allowed to float freely in each subsample, the two-parameter fits yield $R_s = 4188 \pm 841$ AU (high-$|Z|$) and $R_s = 6520 \pm 2740$ AU (midplane), preserving the same ordering ($R_s^{\rm high\text{-}|Z|} < R_s^{\rm midplane}$) albeit with broader uncertainties due to the expected amplitude&ndash;scale degeneracy. The precision asymmetry between the fixed-$\alpha$ results ($\pm 194$ AU for high-$|Z|$ versus $\pm 1573$ AU for the midplane) does not reflect unequal sample sizes&mdash;the two subsamples are comparably populated ($\approx 145{,}000$ and $\approx 141{,}000$ systems respectively)&mdash;but rather the shape of the midplane profile, which rises more gradually and gives the fit less leverage on the onset scale. Despite that broader uncertainty, the ordering is stable across bootstrap realizations and is confirmed by the permutation test below. This is the central environmental signature required by TEP: in the lower-density environment above the thin disk, screening weakens at smaller separation.


This conclusion does not rest only on the metallicity correction formalism. When the analysis is restricted to the solar-track subsample, where the empirical mass-luminosity calibration can be used without any color-dependent correction, the same ordering is recovered: $R_s = 4099 \pm 251$ AU in the high-$|Z|$ subsample and $R_s = 6885 \pm 1223$ AU in the midplane. In both the full sample and the solar-track control, permutation tests ($10{,}000$ realizations each) yield a one-sided significance of $p < 10^{-4}$ for the observed ordering. That replication under an independent control makes the environmental signal much harder to dismiss as a mass-model artifact.


Additional stratified controls indicate that the screening signal is not being generated by a single distance shell or stellar subpopulation. Splitting the full sample at the median distance still yields a finite-transition preference in both halves, with $R_s = 3551 \pm 420$ AU in the nearer systems and $R_s = 10307 \pm 7050$ AU in the more distant half, although the distant subset is much less precise. Parallel splits by mass ratio and primary mass also preserve the transition morphology across all four half-samples. These replications do not replace the midplane&ndash;high-$|Z|$ test, but they do reduce the possibility that the environmental ordering is being driven by one narrow corner of parameter space.


<figure>
  <img src="figures/environment_test.png" alt="Environmental Modulation">
  <figcaption>**Figure 5.1:** Comparison of velocity profiles for binaries in the high-density Galactic midplane ($|Z| < 100$ pc) and the lower-density high-$|Z|$ population ($|Z| > 150$ pc, sampling the thick disk and disk&ndash;halo transition). The lower-density population transitions at smaller separation than the midplane profile, and this ordering is supported by permutation significance in both the full and solar-track analyses.</figcaption>
</figure>

</div>


<div class="discussion">

## 6. Discussion: Resolving the Controversy

The wide-binary debate is often framed as a stark choice: either gravity is universally modified at low acceleration, or the observed signal is primarily artifactual. The TEP framework offers a more specific physical interpretation. It preserves the reality of the anomaly while explaining why the signal should appear with a finite transition scale and a measurable environmental dependence.


### 6.1 Why a Scale-Free MOND Interpretation Is Incomplete
The anomalous velocity boost is detected with high significance. The more discriminating question is whether it is adequately described as a scale-free enhancement, or whether it instead follows a resolved transition in separation. In the present pipeline, the fitted screening profile is favored over a flat Newtonian description by $\Delta \chi^2 = 14{,}845$ and over a separation-independent constant boost by $\Delta \chi^2 = 3{,}583$. The anomaly is therefore not merely an overall low-acceleration excess; it has a finite, separation-structured form. Within TEP, that structure arises naturally because the scalar field remains environmentally suppressed until the binary enters a sufficiently diffuse local regime.


The upgraded model comparison sharpens that conclusion. A sigmoid transition is much worse than the canonical TEP exponential fit. A more flexible double-exponential can reduce the raw $\chi^2$, but it still places the onset at a few thousand AU. The data therefore require a finite transition more robustly than they require one unique phenomenological sharpness. That spread is best treated as model-choice systematic uncertainty, not as evidence for a scale-free MOND-like uplift.


A direct comparison with MOND sharpens this point further. Fitting MOND interpolating functions directly to the binned profile using per-bin median masses (Table 4.1) makes this concrete. Without the External Field Effect (EFE), the simple $\nu$-function recovers an implied $a_0$ consistent with the canonical value, confirming that the MOND transition scale is present in the data, but both variants are catastrophically rejected ($\Delta\chi^2 > +3{,}600$). Incorporating the angle-averaged EFE via QUMOND introduces the saturation that the bare $\nu$-functions lack and reduces the discrepancy substantially, but even the most generous MOND+EFE variant (standard $\nu$, two free parameters) gives $\Delta\chi^2 = +360$ relative to the TEP exponential. The failure is threefold: the best-fit $a_0$ is driven to $5$&ndash;$42\times$ the canonical value, the transition shape is too steep, and the EFE-limited plateau ($\tilde{v} \approx 1.33$) undershoots the observed saturation ($\tilde{v} \approx 1.35$&ndash;$1.40$).


The near-coincidence between $a_0$ and the TEP screening transition is not coincidental in this framework. Smawfield (2025b) showed that the SPARC rotation-curve database yields a characteristic transition acceleration $g_{\rm TEP} \approx 5 \times 10^{-10}$ m/s$^2$, within a factor of four of $a_0$, while Smawfield (2025d) derived $a_0 \sim cH_0$ as an emergent consequence of the scalar field's strong-coupling scale. The distinction between the frameworks is therefore not merely one of scale but of morphology: density-dependent screening produces a gradual, bounded enhancement whose amplitude is set by the local scalar-field value, whereas acceleration-dependent interpolation&mdash;even with the Galactic external field included&mdash;cannot simultaneously reproduce the transition shape, the onset scale, and the saturation level.


### 6.2 The Critical Density Scale
The canonical fit gives a transition separation of $R_s = 2646 \pm 182$ AU, where the quoted error is the formal statistical fit uncertainty; when jackknife stability and model-choice freedom are included conservatively, the total uncertainty broadens to $\pm 563$ AU. For the sample median mass ($M \approx 1.2\,M_\odot$), this transition scale corresponds to an effective binary density $\rho_{eff} \approx 9.2 \times 10^{-18}$ g/cm$^3$. Although the fundamental Universal Critical Density of TEP is much higher ($\rho_c \approx 20$ g/cm$^3$; Smawfield 2025b), the binary sits deep inside the screened Galactic potential. In the schematic notation of Section 2, this is the regime $\rho_{eff} \simeq \rho_{\rm floor} = \epsilon_{\rm env}\rho_c$, with $\epsilon_{\rm env} \sim 4.6 \times 10^{-19}$. As emphasized in Section 2.1, $\epsilon_{\rm env}$ is a post-hoc bookkeeping quantity: any observed $R_s$ can be accommodated by adjusting $\epsilon_{\rm env}$, so the density decomposition alone is descriptive rather than predictive. The observed transition density therefore marks the point at which the binary's internal Newtonian potential becomes sub-dominant to the pre-screened ambient background, not a failure of the universal density scale itself. The genuinely predictive test is the independent cross-check that follows.


The following cross-check constitutes the primary predictive test of the TEP density framework for the wide-binary regime. Smawfield (2025b) derived a characteristic TEP transition acceleration $g_{\rm TEP} \approx 5 \times 10^{-10}$ m/s$^2$ from SPARC rotation-curve fits, entirely independent of the present wide-binary sample. Using the median binary mass at the transition bins ($M \approx 1.3\,M_\odot$), the Newtonian acceleration at the observed $R_s$ is $g_N(R_s) = GM/R_s^2 \approx 1.1 \times 10^{-9}$ m/s$^2$, giving a ratio $\eta = g_N(R_s)/g_{\rm TEP} \approx 2.2$. Equivalently, the zero-free-parameter prediction $R_s^{\rm pred} = \sqrt{GM/g_{\rm TEP}} \approx 3{,}930$ AU overshoots by a factor of 1.5. Part of that residual is quantitatively accounted for by the Galactic external field. At the solar neighborhood, $g_e = v_c^2/R_0 \approx 1.9 \times 10^{-10}$ m/s$^2$; adding this to $g_{\rm TEP}$ as an effective pre-screening floor gives $R_s^{\rm pred} = \sqrt{GM/(g_{\rm TEP} + g_e)} \approx 3{,}340$ AU, reducing the discrepancy to a factor of 1.26. The remaining $\sim 25\%$ offset plausibly reflects the local baryonic disk density, which contributes additional screening beyond the smooth Galactic potential. That the wide-binary and rotation-curve scales agree to within a factor of two, from completely independent datasets and fitting procedures, constitutes a non-trivial consistency test of the TEP screening framework.


### 6.3 Sensitivity to the Mass-Luminosity Relation
The environmental ordering depends critically on the stellar mass estimates. If a solar-metallicity Mass-Luminosity Relation (MLR) is applied uniformly, the masses of metal-poor high-$|Z|$ stars are systematically overestimated, the Keplerian baseline $v_c = \sqrt{GM/s}$ is inflated, and the inferred velocity ratio $\tilde{v}$ in the high-$|Z|$ population is suppressed. Under those conditions the high-$|Z|$ subsample appears to transition at *larger* separation than the midplane, inverting the TEP prediction. The color-dependent MLR correction (Section 3.2) removes this bias by adjusting masses according to each star's offset from the disk color-magnitude ridge, and the corrected analysis recovers the expected ordering: the high-$|Z|$ transition radius is smaller than the midplane value for both the full sample ($R_s = 4673 \pm 194$ AU versus $7159 \pm 1573$ AU) and the solar-track control ($R_s = 4099 \pm 251$ AU versus $6885 \pm 1223$ AU; permutation $p < 10^{-4}$ in both cases). That the solar-track control, which uses an empirical mass calibration with no color-dependent correction at all, independently recovers the same ordering makes the result difficult to dismiss as an MLR artifact.


The supplemental controls narrow the most common non-TEP interpretations still further. If the observed profile were driven mainly by one demographic subset, the transition would be expected to collapse under stratification by mass ratio, primary mass, or observational geometry. Instead, the signal persists across all four stellar-population half-samples, with each remaining strongly preferred over a constant-boost alternative, and it is still recovered in a stricter radial-velocity consistency subset of 6117 systems, though with a larger central $R_s$ and much broader uncertainty owing to the $\sim 55\times$ smaller sample size. Conversely, when the catalog-level labels are scrambled, either globally or within distance quartiles, none of $10{,}000$ valid realizations reproduces the observed screening preference ($p < 10^{-4}$ in both cases). The most straightforward reading of these controls is therefore that the Gaia DR3 anomaly is structured, population-robust, and difficult to reduce to a single calibration artifact.


### 6.4 Limitations and Outlook

Several caveats deserve explicit acknowledgment. First, the reduced $\chi^2_{\nu} = 5.1$ for the canonical fit indicates that the bin-level bootstrap errors underestimate the true scatter, probably because residual substructure in the Gaia sample (e.g., moving groups, distance-correlated completeness) introduces correlated fluctuations. The morphological adequacy of the fit is better assessed by the runs test ($p = 0.485$) and by the consistency of $R_s$ across demographic splits. As shown in Section 4, inflating the bin errors by $\sqrt{\chi^2_{\nu}}$ to force $\chi^2_{\nu} = 1$ preserves all model-comparison conclusions, but a more refined error model would be desirable.


Second, the null-control and environmental permutation tests use $10{,}000$ realizations each, giving a $p$-value resolution floor of $10^{-4}$. No realization in either ensemble reproduced the observed $\Delta\chi^2$ or environmental ordering ($p < 10^{-4}$ in all cases).


Third, an injection-recovery test validates that the profile-fitting pipeline faithfully recovers a known screening signal and does not hallucinate one when none is present. One hundred mock catalogs were constructed by globally shuffling the observed $\tilde{v}$ values (destroying all separation-dependent structure) and then injecting a TEP enhancement with the observed parameters ($R_s = 2646$ AU, $\alpha_{\rm sat} = 0.366$). The pipeline recovers $R_s = 3{,}005 \pm 334$ AU (median $\pm$ scatter) and $\alpha_{\rm sat} = 0.346 \pm 0.014$, with a modest positive bias of $\sim 14\%$ in $R_s$ attributable to the median-based binning procedure, and $100\%$ detection rate ($\Delta\chi^2 > 10$ versus a constant boost in every realization). Crucially, when the same procedure is repeated with zero injected signal ($\alpha = 0$), the pipeline returns $\alpha \approx 0$ and a $0\%$ false-positive rate, confirming that the analysis does not manufacture a transition from featureless data. A sweep across six injected $R_s$ values ($1{,}000$&ndash;$8{,}000$ AU) shows monotonically faithful recovery with $100\%$ detection at every scale, demonstrating that the pipeline is sensitive to screening transitions across the full range probed by the data.


Fourth, the observable $\tilde{v}$ is a projected quantity. As discussed in Section 3.3, the normalization procedure cancels the leading geometric factor for a thermal eccentricity distribution, so the inferred transition radius $R_s$ is robust. The absolute saturation amplitude $\alpha_{\rm sat}$, however, carries an unquantified systematic at the few-percent level from the unknown true eccentricity distribution.


Fifth, angular resolution sets a distance-dependent floor on the minimum resolvable projected separation. Wide pairs at large heliocentric distance are preferentially lost at small $s$, which could in principle create a distance-separation correlation that mimics a transition. The distance-split control (Section 5) mitigates this concern: both the near and far halves independently prefer a finite screening transition, and the near subsample, which is least affected by resolution, yields a transition scale ($R_s = 3551 \pm 420$ AU) consistent with the full-sample value.


Sixth, the MLR correction uses a linear color-offset model with a single coefficient $\beta_{\rm MLR} \approx 1.5$. A quadratic or nonparametric correction could absorb additional metallicity structure; however, the solar-track control, which bypasses the MLR correction entirely, reproduces both the transition and the environmental ordering, suggesting that the linear approximation is adequate for the present analysis.


Seventh, the MOND comparison (Table 4.1) uses per-bin median masses and incorporates the angle-averaged EFE via QUMOND. Remaining refinements include marginalizing over the full mass distribution within each bin (rather than using the median), testing alternative EFE geometries, and exploring non-standard interpolating functions that might accommodate saturation more naturally. These would reduce but are unlikely to eliminate the $\Delta\chi^2 = +360$ discrepancy of the best MOND+EFE variant, given the threefold failure in acceleration scale, shape, and amplitude.


Eighth, several physical quantities imported from other papers in the TEP series&mdash;the universal critical density $\rho_c \approx 20$ g/cm$^3$ and the characteristic acceleration $g_{\rm TEP} \approx 5 \times 10^{-10}$ m/s$^2$ (Smawfield 2025b), and the emergent MOND-scale derivation $a_0 \sim cH_0$ (Smawfield 2025d)&mdash;are drawn from preprints that have not yet undergone independent peer review. The present analysis does not depend on the precise numerical values of $\rho_c$ or $g_{\rm TEP}$ for any of its primary results (the screening fit, environmental ordering, and MOND comparison are all internal to the Gaia data); those quantities enter only in the cross-scale consistency check of Section 6.2. Readers should regard that check as a promising but provisional link pending independent verification of the input values.


Ninth, the environmental comparison fixes $\alpha_{\rm sat} = 0.4$ to break the amplitude&ndash;scale degeneracy (Section 5.1). When $\alpha_{\rm sat}$ is instead allowed to float freely, the two-parameter fits yield $R_s = 4188 \pm 841$ AU (high-$|Z|$) and $R_s = 6520 \pm 2740$ AU (midplane), preserving the ordering $R_s^{\rm high\text{-}|Z|} < R_s^{\rm midplane}$ with broader uncertainties. The fixed-$\alpha$ protocol is therefore a precision choice, not one that creates the environmental signal.


Tenth, at the shortest separations ($s \lesssim 100$ AU), unresolved spectroscopic binaries can inflate photometric mass estimates without proportionally affecting the proper-motion difference, potentially deflating the inner-baseline $\tilde{v}$ and inflating the apparent outer-bin enhancement. As discussed in Section 3.3, the RUWE cut and five-bin normalization window mitigate this effect, and excluding Bin 1 shifts $R_s$ by less than $2\%$.


Looking ahead, Gaia DR4 will provide improved astrometry, additional radial velocities, and a longer time baseline, enabling tighter proper-motion uncertainties and more complete rejection of hierarchical triples. A quantitative prediction from the present framework is that the transition radius should shift systematically with Galactocentric radius and with local stellar density, providing a further falsifiable test of the TEP screening mechanism.


</div>


<div class="conclusion">

## 7. Conclusion

The Gaia DR3 wide-binary population provides strong empirical support for the density-dependent screening mechanism predicted by the Temporal Equivalence Principle (TEP). From a high-purity sample of 341,315 systems, the analysis identifies a characteristic transition radius of $R_s = 2646 \pm 182$ AU (statistical) from the canonical TEP exponential fit. A conservative uncertainty budget that includes jackknife stability and model-choice freedom broadens the total uncertainty to $\pm 563$ AU while preserving the same few-thousand-AU onset scale.


The transition corresponds to an effective local binary density of $\rho_{eff} \approx 9.2 \times 10^{-18}$ g/cm$^3$, far below the Universal Critical Density ($\rho_c \approx 20$ g/cm$^3$; Smawfield 2025b). As discussed in Section 6.2, this gap is expected: the conformal scalar field is already heavily screened by the Galactic halo and baryonic disk, and the observed transition marks the point where the binary's internal potential falls below the local residual screening floor. An independent cross-check reinforces this: the TEP characteristic acceleration $g_{\rm TEP} \approx 5 \times 10^{-10}$ m/s$^2$ derived from SPARC rotation curves (Smawfield 2025b) predicts $R_s^{\rm pred} \approx 3{,}930$ AU; when the Galactic external field ($g_e \approx 1.9 \times 10^{-10}$ m/s$^2$) is included as an additional screening floor, the prediction tightens to $\approx 3{,}340$ AU, within a factor of 1.26 of the observed value. The fitted screening profile is favored over a separation-independent constant boost by $\Delta \chi^2 = 3{,}583$ (Table 4.1), and alternative smooth-transition fits preserve the finite onset scale, confirming that the transition is not an artifact of the canonical functional form. Direct comparison with MOND interpolating functions, including the angle-averaged External Field Effect with per-bin median masses, shows that even the most generous MOND+EFE variant ($k=2$) is rejected by $\Delta\chi^2 = +360$ relative to TEP, failing simultaneously in transition shape, onset scale, and saturation amplitude.


The environmental test provides independent corroboration. The analysis shows that the inferred ordering between midplane and high-$|Z|$ subsamples is sensitive to the mass-luminosity relation: a uniform solar-metallicity MLR overestimates halo-star masses and inverts the ordering, whereas the color-dependent correction (Section 3.2) and an independent solar-track calibration both recover the TEP-predicted pattern. With those corrections in place, the lower-density high-$|Z|$ population enters the anomalous regime at smaller radius than the higher-density midplane, both in the full sample ($R_s = 4673 \pm 194$ AU versus $7159 \pm 1573$ AU) and in the solar-track control ($R_s = 4099 \pm 251$ AU versus $6885 \pm 1223$ AU; permutation $p < 10^{-4}$ in both cases). These results provide both a coherent screening-scale signal and an independent environmental verification of the same underlying TEP mechanism.


Supplemental controls strengthen that interpretation still further. The transition remains present after stratifying the catalog by distance, mass ratio, and primary mass, with each stellar-population half-sample continuing to favor a finite screening profile over a constant boost. A stricter radial-velocity consistency subset also preserves the effect, although with broader uncertainty because of its smaller sample size. Most importantly, catalog-level scrambling tests fail to reproduce the observed screening preference: neither a global scramble nor a distance-preserving scramble matches the measured improvement over the null descriptions in $10{,}000$ valid realizations ($p < 10^{-4}$ in both cases). An injection-recovery test further validates the pipeline: it recovers a known TEP signal with $100\%$ detection rate and returns a $0\%$ false-positive rate when no signal is injected. These controls do not generate the signal, but they make it appreciably harder to attribute the result to a narrow demographic selection, a trivial catalog artifact, or a pipeline bias.


The wide-binary anomaly therefore emerges not as a generic failure of dark matter or standard inertia, but as a structured, environmentally modulated screening transition whose morphology, onset scale, and density dependence are quantitatively consistent with the conformal scalar field of TEP and are not reproduced by MOND with or without the External Field Effect. The present analysis establishes the wide-binary regime as a new, independent probe of the TEP screening mechanism, complementing the rotation-curve and compact-object evidence presented elsewhere in the series.


</div>


<div class="references">
    ## References

    <p class="reference-item">Banik, I., Pittordis, C., Sutherland, W., Famaey, B., Ibata, R., Mieske, S., &amp; Zhao, H. 2024, MNRAS, 527, 4573. *Strong constraints on the gravitational law from Gaia DR3 wide binaries.*</p>

    <p class="reference-item">Burrage, C. &amp; Sakstein, J. 2018, Living Reviews in Relativity, 21, 1. *Tests of Chameleon Gravity.*</p>

    <p class="reference-item">Chae, K.-H. 2023, ApJ, 952, 128. *Breakdown of the Newton–Einstein Standard Gravity at Low Acceleration in Internal Dynamics of Wide Binary Stars.*</p>
    
    <p class="reference-item">Chae, K.-H. 2024a, ApJ, 960, 114. *Robust Evidence for the Breakdown of Standard Gravity at Low Acceleration from Statistically Pure Binaries Free of Hidden Companions.*</p>

    <p class="reference-item">Chae, K.-H. 2024b, ApJ, 968, 103. *Rebuttal of Claimed Severe Flaws in the Breakdown of Standard Gravity: A Reply to Banik et al.*</p>

    <p class="reference-item">El-Badry, K., Rix, H.-W., &amp; Heintz, T. M. 2021, MNRAS, 506, 2269. *A million binaries from Gaia eDR3: sample selection and validation of Gaia parallax uncertainties.*</p>

    <p class="reference-item">Famaey, B. &amp; Binney, J. 2005, MNRAS, 363, 603. *Modified Newtonian dynamics in the Milky Way.*</p>

    <p class="reference-item">Gaia Collaboration, Vallenari, A., Brown, A. G. A., et al. 2023, A&amp;A, 674, A1. *Gaia Data Release 3. Summary of the content and survey properties.*</p>

    <p class="reference-item">Hernandez, X. 2023, MNRAS, 525, 1401. *Internal kinematics of Gaia DR3 wide binaries: anomalous behaviour in the low acceleration regime.*</p>

    <p class="reference-item">Hernandez, X. &amp; Cort&eacute;s, R. A. M. 2024, MNRAS, 529, 4988. *Wide binaries from Gaia DR3: signal persistence under quality cuts and distance stratification.*</p>

    <p class="reference-item">Khoury, J. &amp; Weltman, A. 2004, Physical Review Letters, 93, 171104. *Chameleon Fields: Awaiting Surprises for Tests of Gravity in Space.*</p>

    <p class="reference-item">Milgrom, M. 1983, ApJ, 270, 365. *A modification of the Newtonian dynamics as a possible alternative to the hidden mass hypothesis.*</p>

    <p class="reference-item">Pe&ntilde;arrubia, J. 2024, MNRAS, 532, 1721. *Tidal streams as probes of gravity in the low-acceleration regime.*</p>

    <p class="reference-item">Pittordis, C., Sutherland, W., &amp; Shepherd, P. 2025, Open Journal of Astrophysics, 8. *Wide Binaries from Gaia DR3: testing GR vs MOND with realistic triple modelling.* DOI: 10.33232/001c.142887.</p>

    <p class="reference-item">Smawfield, M. L. 2025b, Zenodo. *Universal Critical Density: Unifying Atomic, Galactic, and Compact Object Scales.* (TEP Series, Paper 7.) DOI: 10.5281/zenodo.18029598.</p>

    <p class="reference-item">Smawfield, M. L. 2025c, Zenodo. *The Temporal Equivalence Principle: Dynamic Time, Emergent Light Speed, and a Two-Metric Geometry of Measurement.* (TEP Series, Paper 1.) DOI: 10.5281/zenodo.16921911.</p>

    <p class="reference-item">Smawfield, M. L. 2025d, Zenodo. *Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations.* (TEP Series, Paper 5.) DOI: 10.5281/zenodo.17982540.</p>
</div>


