# bi-PINN
In the rapidly evolving field of scientific machine learning, physics-informed neural networks (PINNs) have
emerged as a powerful paradigm for numerically solving partial differential equations (PDEs). Within this
framework, boundary conditions serve as essential physical prior knowledge. However, the approximate
enforcement of these conditions often emerges as a critical accuracy bottleneck, particularly near the domain
boundaries. To address this challenge, we propose a boundary-information-enhanced physics-informed neural
network (bi-PINN), which systematically strengthens the impact of boundary physics on the training process.
The core innovation lies in a soft boundary-derivative restraint (bd-restraint) that incorporates higher-order
or tangential derivative information—derived from the underlying physical laws or the differential-geometric
properties of the boundary manifold—directly into the loss function. Since these derivative relationships are
typically not explicit in the raw boundary data, the bd-restraint is generated offline via symbolic or automatic
differentiation of the governing equations and the geometric description. We validate the effectiveness of
the bi-PINN through various benchmark problems that span diverse physical regimes: the Poisson equation,
linear elasticity, heat conduction problem, and the Korteweg–de Vries (KdV) equation. Extensive numerical
comparisons demonstrate that the bi-PINN achieves a one to two-order of magnitude reduction in boundary
error and consistently improves accuracy across the entire computational domain compared to conventional
PINNs. These results underscore the potential of bi-PINN for high-fidelity simulations of complex and freeboundary
problems, especially when only partial physical knowledge is available. These results underscore
the computational efficiency and potential of bi-PINN in tackling real-world and complex problems.

If you have any qusetions about the paper or the code, please contact my email. 
I'm happy to discuss any academic questions with you!
My email is: 79620664@qq.com.

### MATLAB Requirements
- **Required Version**: MATLAB R2023a or higher
- **Critical Note**: Versions older than R2023a are incompatible and will not run correctly
