#import "@preview/lovelace:0.3.0": *
#import "@preview/blinky:0.1.0": link-bib-urls

#show figure.caption: set align(start);
#show figure.caption: set text(8pt)
#show heading: set text(18pt)
#show heading: set block(spacing: 1.5em)
#set list(indent: 1em)
#set text(font: "New Computer Modern Sans")
#show cite: set text(style: "italic")
#show cite: set highlight(fill: red)
#set par(justify: true)
#set page(paper: "a4")

#set par(leading: 1em) //line space

#show cite: it => {
  // color everthing except brackets
  show regex("[a-zA-Zöü&\d.,.-]"): set text(fill: blue)
  // or regex("[\p{L}\d+]+") when using the alpha-numerical style
  it
}

#show ref: it => {
  if it.element == none {
    // This is a citation, which is handled above.
    return it
  }

  show regex("[a-zA-Zöü&\d.,.-]"): set text(fill: blue)
  it
}

#show outline.entry.where(
  level: 1,
): it => {
  v(12pt, weak: true)
  strong(it)
}

#let left_sign = rotate(
    -90deg,
    place(
      dx: -450pt,
      dy: -37pt,
      [Synthesized sensor data from Neural Radiance Fields - Patrick Kaserer],
    )
  )

#let right_sign = rotate(
    -90deg,
    place(
      dx: -450pt,
      dy: 490pt,
      [Synthesized sensor data from Neural Radiance Fields - Patrick Kaserer],
    )
  )
  
// ------------------------------------------------------------------- TITLE PAGE

#counter(page).update(0)
#set page(numbering: "I")
#set page(footer: none)

#place(
  dx: (100% - 180pt),
  image("images/Hochschule_Furtwangen_HFU_logo.svg", width: 200pt)
)

#place(
  dy: 7pt,
  dx: 0pt,
  image("images/Logo_SICK_AG_2009.png", width: 150pt)
)
#v(100pt)
#strong(text([Master Thesis], size: 45pt, baseline: 20pt))

#text([by], size: 15pt, baseline: 8pt) \

#strong(text([Patrick Kaserer, B.Sc], size: 20pt))

#text([This thesis submitted for the degree of], size: 15pt, baseline: 17pt) \

#strong(text([Master of Science (M.Sc)], size: 20pt, baseline: 10pt))

#text([on the faculty of digital media], size: 15pt) \
#v(35pt)
#strong(text([Synthesized Sensor Data from], size: 25pt, spacing: 18pt)) \

#strong(text([Neural Radiance Fields], size: 25pt, spacing: 18pt)) \

#v(40pt)

#let title_size = 15pt
#place(
  dy: 25pt,
      table(
          columns: 2,
          stroke: none,
          align: left,
          inset: 6pt,
          text([Degree course:], size: title_size), text([Computer Science in Media, Master], size: title_size),
          text([Matriculation number:], size: title_size), text([272509], size: title_size),
          [],[],
          [],[],
          [],[],
          text([University supervisor:], size: title_size), text([Prof. Dr. Uwe Hahne], size: title_size),
          text([External supervisor:], size: title_size), text([Dr. Jakob Lindinger], size: title_size),
          text([submitted], size: title_size), text([30.11.2024], size: title_size),
  )
)

#pagebreak()

= Abstract
#pagebreak()

= Zusammenfassung
#pagebreak()

= Preface
#pagebreak()

= Table of Contents
#outline(depth: 3, indent: 2em, title: none)
#set heading(numbering: "1.")
#set page(numbering: "1")
#counter(page).update(1)

#context counter(heading).update(0)
// #pagebreak()
#set page(footer: auto)

= Introduction <introduction>
#left_sign
Sensors enable machines to perceive and interpret the world around them. By allowing humans, through machines, to observe and measure aspects of the environment beyond human capability, sensors foster the development of technologies and machines that enhance our understanding of the world, physics, human behavior, and more. They are, therefore, indispensable for technological advancement.

This thesis explores whether a synthetic LiDAR sensor can be generated within a Neural Radiance Field (NeRF) framework. By simulating LiDAR sensors in a NeRF, the aim is to determine the feasibility of virtual sensor models, offering insights into which physical sensors may be suitable or necessary for real-world applications. It explores the advantages and disadvantages of this approach.

Industrial applications can be dangerous. The use of inappropriate sensors can cause injuries or worse. Testing potential sensors in a virtual environment, such as a NeRF, is safer and more efficient than experimenting with real sensors. Therefore, developing a method to simulate LiDAR sensors within NeRFs could significantly aid in selecting the correct sensors for specific applications. To understand how a LiDAR sensor can be simulated within a NeRF, it is essential to delve into the principles of distance measurement and the role of constants like the velocity of light. 

An important area for sensors is measuring, especially distance measuring. In order to facilitate measurement, it is necessary to employ a constant value that is universally applicable and accessible to all. One such fundamental physical constant is the velocity of light, which can be used to calculate the distance between two points in space by determining the time required for light to travel between them. This concept forms the basis of the measurement principle known as "time-of-flight" (ToF) #cite(<hahne_real-time_2012>, supplement: [p.~11]). One prominent example of a time-of-flight sensor is Light Detection and Ranging (LiDAR). They are different types of LiDAR. One type of LiDAR–which is the focus of this study—is the use of lasers that are directed at an object and then measure time it takes for the reflected light to return to the receiver, allowing it to generate a 3D or 2D image with spatial and depth data @noauthor_sick-lidar_nodate. It is difficult to estimate which LiDAR could be correct for each given application, highlighting the need for effective simulation methods. By focusing on simulating LiDAR sensors within a NeRF, this thesis aims to provide a tool for assessing the suitability of various LiDAR technologies in a controlled virtual environment only with images from the scene.

This approach could offer significant advantages in terms of safety, cost, and efficiency when determining the most appropriate sensor for specific industrial applications. To effectively simulate LiDAR sensors in a virtual environment, advanced methods for 3D scene representation are required.

It is difficult to measure the real world. Cameras can help, but they only show 2D scenes. Using a LiDAR sensor to measure the real world creates a 3D point cloud what is more difficult to understand. One way to create a 3D scene from 2D images is with NeRF.

In 2020, #cite(<mildenhall_nerf_2020>, form: "author") introduced NeRF as an AI-based approach for generating 3D representations of scenes from 2D images. One of the key advantages by using NeRF is the low memory usage and the photo-realistic representation. With the use of pipelines, it is also easy to use NeRFs, even for untrained users @schleise_automated_2024. The central idea behind NeRF is to utilize a neural network, rather than techniques that use a grid or a list system to determinate each point and its corresponding values within a 3D scene.
#right_sign
Like other neural networks, NeRF suffers from the disadvantage of being a "black box" @la_rocca_opening_2022. This makes it difficult or even impossible to fully understand the computations taking place, as the neural network provides limited interpretability. As a result, it is challenging to determine positions, distances, or lengths within the scene. Unlike methods, which typically store scene data in grids or lists, where each pixel within the scene is explicit know, NeRF only compute RGB and density values from given coordinates. Depending on the method which the neural network is trained, the distances and coordinates from the original scene are unknown.\

In other graphical software, such as Unity, NVIDIA Omniverse, and Blender, the coordinates of each edge, vertex, or face are explicitly defined. However, in NeRF, these coordinates are not known. To illustrate this concept in a simplified manner, each scene in NeRF can be conceptualized as a "memory" of the neural network, which learns to generate RGB and density values from a multitude of perspectives and positions by minimizing the discrepancy between the original image and its internal representation.

This implementation can be employed for the purpose of measuring distances within a given scene, in a manner analogous to that of an artificial LiDAR. Additionally, it can be utilized for the precise measurement of distances between two points within the scene. Furthermore, the implementation allows for the use of a reference object within the scene, which is used to compute a scale factor that is then applied to the measured distances. This method also can used to mapping closed areas like an LiDAR sensor. \
In order to implement the method, a technique is employed which enables the detection of the density value at a given point within the scene, as well as the calculation of the distance between the origin and the points. NVIDIA Omniverse is utilized for the generation of images, the management of camera parameters and as a means of comparison with a simulated LiDAR. Meanwhile, Nerfstudio with Nerfacto is employed for the creation of the NeRF and the implementation of the proposed methods.
#pagebreak()

= Related Work <related-work>
To understand the implementation and the associated challenges, this chapter provides a deeper dive into the fundamental principles of Neural Radiance Fields (NeRF). Additionally, it highlights similar works that combine NeRF with LiDAR systems, offering a comprehensive context for the methods and contributions discussed later in this thesis.

== Neural Radiance Fields <neural-radiance-fields>
#cite(<mildenhall_nerf_2020>, form: "author") present a method for generating an AI-based 3D scene representation called Neural Radiance Fields (NeRF). The key idea behind NeRF is to synthesize novel views from limited input. The advantage of using a neural network is that it requires less memory than techniques that use lists and produces photo-realistic results. The main component is a neural network $F theta$, implemented as a multilayer perceptron (MLP), which is trained using the positions and directions (perspectives of pixel in an images) from input images to approximate RGB values and compute the density $sigma$ at a given point in the scene. The density and RGB values are then used to generate a 3D scene through volumetric rendering.
#v(3mm)
#figure(image("images/nerf_image_mildenhall_1.png", width: 85%), caption: [Overview of the NeRF scene representation. Sampling the 5D input location and direction $x, y, z, theta, phi$ (a) to feed the neural network (b). Then using volumetric rendering techniques to render the scene (c) and optimizing the scene with backpropagation since volumetric rendering
function is differentiable (based on Mildenhall et al. (2020), Figure 2).])<fig:shift_image>
#v(3mm)
One problems which NeRF address is  the ill-posed problem:

- #strong[Non-unique solution:] There is no clear way to reconstruct a 3D scene from limited 2D images.

- #strong[Sensitivity:] Small changes in the input can have big changes in the output.

- #strong[Incompleteness of the data:] Not all information about the scene is known in the input data.

NeRF address this ill-posed problem by utilizing a neural network that learns contextual information from the scene. The neural network minimizes the loss function between its computed outputs and the ground truth images, thereby learning to approximate the original scene. This approach enables reconstruction even from perspectives for which no images are provided, as long as spatial information is available from other images.

=== Positioning and parameters <positioning>
An important part of this study is the positioning and parameters for the NeRF to train the model. These are responsible for distance measurement and scene scale within the NeRF scene. NeRF learns from images. To do this, most NeRF variants require the extrinsic and intrinsic camera parameters. These parameters are important for measuring distances in a NeRF scene because both the positions and the scale factor in NeRF depend on the image parameters. \
These parameters are:
- The exact positions and orientation of each images used for the scene in form of a homogeneous 4x4 transform matrix: $ mat(
  delim: "[",
  r_11, r_12, r_13, p_x;r_21, r_22, r_23, p_y;r_31, r_32, r_33, p_z;0, 0, 0, 1,
) $

- #strong[Image Resolution:] Must be the same for all images when using Nerfstudio.\
- #strong[Focal Length:] Determining the optical distance between the sensor at the camera's focal plane and the location where light rays converge to create a clear image of an object. The angle of view is wider and the magnification is lower with a shorter focal length @black_nikonusa_nodate.
- #strong[Principle Point:] The principal point represents a fundamental concept in the fields of optics and photography. It denotes the point on the image plane at which the optical axis of the lens intersects the vertical axis. In theoretical models, this point is situated at the geometric center of the image sensor or film. However, in practical applications, manufacturing tolerances or lens aberrations can result in a slight deviation of the principal point from its intended position @clarke_principal_1998. 
- #strong[Distortion:] Is an optical aberration in which straight lines in the object appear as curved lines in the image. This effect often occurs with lenses and can affect the geometric accuracy of images. Prominent forms are barrel distortion and pincushion distortion @li_efficient_2024.

In an ideal case, as in @fig:pos, all cameras are evenly spaced around a center point and all camera positions are known. This center point thus acts as the center of gravity and the origin of the coordinate system of the NeRF scene. The NeRF architecture recognizes and learns from the camera positions and can derive the distances in the scene based on these positions.\

#figure(image("images/related_work/pose.png"), caption: [Left-handed: Symmetric and known camera positions where the centroid is in the center of the scene. The result would be the same coordinate system with the same distance in NeRF as in the original system. Right-handed: In order to simulate unknown camera positions,  the value of $x+1$ is added to each camera position. It should be noted that the centroid remains the midpoint of all positions, while the origin from the NeRF-Scene is situated at the top, with the coordinates $(0,0,0)$.])<fig:pos>

For example, if the cameras are exactly one meter away from the center point, this distance is also correctly interpreted by the NeRF and used to reconstruct the geometry of the scene. In practice, however, symmetry or known camera poses are rare because the original scene does not allow symmetric image creation or the technical availability is not given for image poses outside a virtual world. Without knowing the exact camera parameters, this is often determined using tools that use structure-for-motion methods such as colmap, which leads to inaccuracies in pose detection. Using exact coordinates is faster and more precise than using tools for pose estimation and a reference object within the scene for scale estimation. \
Training the model with known camera poses has the benefit that the model knows exact positions from the original scene and it reduces the possibility of making a calibration error with a reference object within the scene. Navigation and usability in the scene are also easier as the original coordinates can be used. \

#figure(image("images/related_work/pose2.png"), caption: [Comparison of the two NeRF scene with same original scene and images but different parameter creation. While the origin in the scene with known parameters is the same as in the original scene (A), the scale as well as the position and rotation are different in the scene where the parameters are estimated with colmap (B) to see at the frustum location and orientation within the image. Note: The scale of the frustum in B is larger than in A, the frustum is not closer to the scene camera.])<fig:pos3>

This is illustrated in @fig:pos3. While the coordinates in the scene with known parameters are approximately identical to those in the original scene, the use of colmap for parameter estimation results in a shift of these coordinates. Additionally, the scale differs, resulting in erroneous values when measuring distances without reference object.\
\
#strong[Structure from Motion]\
Colmap is used for pose estimation with structure from motion (SfM). Since Structure from Motion is based on the reconstruction of 3D point positions and camera parameters exclusively from 2D images, the resulting scene geometry is only defined on a relative scale. This means that neither the absolute scaling nor the exact coordinates of the original scene can be determined without additional reference information.

#v(4mm)

#strong[Pose Estimation with Marker]\
Using marker for pose estimation 

=== Neural Network Architecture <neural-network-architecture>
The MLP consists of nine fully connected layers: eight layers with 256 neurons each and one with 128 neurons. It uses the #emph[Rectified Linear Unit] (ReLU) as the activation function, a non-linear function that leaves positive inputs unchanged while setting negative inputs to zero.

After capturing multiple images of a scene from different perspectives and obtaining the extrinsic and intrinsic camera parameters, the neural network uses the pixels from each image to cast rays into the scene. These rays are divided into samples (points) at specific intervals. Each point has its own coordinates in the scene and a direction derived from the ray.

While the densities $sigma$ are determined by the exact position in space, the RGB values $c$ also requires the direction $theta , phi.alt$. This is important for multi-view representation because viewing the same position from different angles can reveal different content. This technique allows for the correct rendering of reflections and lighting effects depending on the perspective. To achieve this, the first eight layers of the MLP process the 3D coordinates $X = x , y , z$, outputting a density value and a feature vector. This feature vector is then combined with the viewing direction as polar coordinates $theta , phi.alt$ (the pixel perspective), and passed into the final layer, which returns normalized RGB color values @mildenhall_nerf_2020. Formally, this function can be represented as: \
\
$ X , theta , phi.alt arrow.r R G B , sigma $ \
This function is important for solving a challenge of this thesis. To estimate the point on an object, a method is needed to calculate the correct density when the point is as close as possible to the obstacle.
#figure(image("images/related_work/nerf_rays.png", width: 60%), caption: [Illustration of rays in NeRF. Casting a ray for each pixel from an image, which includes the 5D input direction $theta, phi$ and sampling points as a coordinate $x, y, z$. The direction and the coordinates are the input for the neural network, which outputs the density $sigma$ and the RGB value.])<fig:nerf_rays>

=== Volume Sampling in Neural Radiance Fields <volume_sampling>
A critical component of NeRFs is the sampling strategy employed to evaluate the radiance field along camera rays. This section discusses the stratified sampling approach used in NeRFs, explaining how it selects sample points along rays, computes transmittance to detect potential object intersections, retrieves RGB and density values, and ultimately renders the scene from a chosen perspective.\
#v(3mm)
#strong[Stratified Sampling in NeRFs]

To render a scene using NeRF, we need to estimate the expected color $C(r)$ of each camera ray $r$ passing through every pixel of the virtual camera. The ray is parameterized as $r(t)=o+t d$, where $o$ is the camera origin, $d$ is the direction of the ray, and $t$ ranges from near bound $t_n$ ​ to far bound $t_f$.
The continuous integral for the expected color along a ray is given by:\
$ C(r) = integral^(t_f)_(t_n) bold(T)(t)bold(sigma)(r(t)) bold(c)(r(t),d)d t, $

where:
- $bold(sigma)(r(t))$ is the volume density at point $r(t)$,
- $bold(c)(r(t),d)$ is the emitted radiance (color) at point $r(t)$ in direction $d$,
- $bold(T)(t) = exp(- integral_(t_n)^t sigma (r(s))d s)$ is the accumulated transmittance from $t_n$ to $t$, representing the probability that the ray travels from $t_n$ to $t$ without interaction.\
#v(5mm)
#strong[Numerical Approximation Using Stratified Sampling]\
#v(3mm)
Computing the integral analytically is intractable due to the complexity of the scene and the continuous nature of the radiance field. Therefore, NeRF approximate the integral numerically using stratified sampling combined with quadrature.

- #strong[Partitioning the Ray interval:] The interval $[t_n, t_f]$ is divided into $N$ evenly spaced segments (strata). Each segment corresponds to a "bin" along the ray.

- #strong[Random sampling within strata:] Within each bin, a single sample point $t_i$ is randomly selected using a uniform distribution:

$ t_i ∼ U[t_n + (i-1)/N (t_f-t_n), t_n+i/N (t_f - t_n)] $

where $U[a,b]$ denotes a uniform distribution between $a$ and $b$, and $i ∈ {1,2 , ..., N}$

By dividing the interval into equal segments, stratified sampling ensures that samples are distributed across the entire ray, preventing clustering in any specific region. Random sampling within each stratum reduce the variance of the integral approximation compared to purely random sampling.
#v(5mm)
#strong[Estimating the Color Integral]\
#v(3mm)
Using the sampled points, the integral is approximated with a quadrature sum:
$ accent(C,"^")(r) = sum_(i=1)^N T_i alpha_i c_i, $

where:
- $T_i = exp(-sum_(j=1)^(i-1) sigma_j delta_j)$ is the accumulated transmittance up to the $i$-th sample,
- $alpha_i = 1 - exp(-sigma_i delta_j)$ represents the opacity at the $i$-th sample,
- $c_i = c(r(t_i),d)$ is the color remitted at $t_i$,
- $delta_i = t_(i+1) - t_i$ is the distance between adjacent samples along the ray.

#v(5mm)
#strong[Computing Transmittance and Detecting Collisions]\
#v(3mm)
The transmittance $T_i$ accounts for the attenuation of light due to the accumulated density along the ray. A higher density $sigma_i$ at a point indicates a higher probability of the ray interacting with matter (e.g., hitting an object). By computing $T_i$ and $alpha_i$, the algorithm effectively determines the regions along the ray where collisions (interactions) are likely to occur.

#v(5mm)
#strong[Retrieving RGB and Density Values]\
#v(3mm)
At each sampled point $t_i$ the neural network is queried to obtain the emitted color $c_i$ and the volume density $sigma_i$. These values are the used in the quadrature sum to estimate the expected color $accent(C, "^")(r)$ of the ray.
 
#v(5mm)
#strong[Rendering the scene]\
#v(3mm)

The estimated colors $accent(C, "^")(r)$ for all rays passing through the camera's image plane are aggregated to render the final image from the desired perspective. This process accounts for the volumetric effects and occlusions in the scene, resulting in a realistic rendering.

=== Hierarchical Volume Sampling

Hierarchical volume sampling is an advanced sampling strategy designed to allocate computational resources efficiently by focusing on regions along the ray that are more likely to contribute to the final rendering. It involves two key stages:

- #strong[Coarse Sampling Stage:] A set of $N_c$ sample points ${t_i}$ is drawn along the ray using stratified sampling, as described in the previous section. A simpler neural network (the "coarse" network) evaluates the density $sigma_i$ and color $c_i$ at each sample point $t_i$. The coarse network than provides an initial estimate of the transmittance an opacity along the ray.

- #strong[Fine Sampling Stage:] Base on the outputs from the coarse network, a probability density function (PDF) is constructed to reflect the likelihood of each region along the ray contributing to the final color. A Second set of $N_f$ sample points is drawn from this PDF using inverse transform sampling, concentrating samples in regions with higher estimated density. A more detailed neural network (the "fine" network) evaluates the density and color at the combined set of $N_c + N_f$ sample points.

In regions where the coarse network indicates higher density (i.e., potential object surfaces), the sampling density is increased during the fine sampling stage. By allocating more samples to these critical regions, the algorithm achieves higher accuracy in distance measurement and rendering details. This is important for detecting an object in the scene and thus for distance measurement.

#v(5mm)
#strong[Constructing the PDF for Importance Sampling]\
#v(3mm)

The weights $w_i$ from the coarse network's output are used to create the PDF:
$ w_i = T_i (1 - exp(-sigma_i delta_i)) $
where $T_i$ and $delta_i$ are as previously defined. The weights are normalized to form a valid PDF. This PDF guides the final sampling stage, ensuring that more samples are drawn in regions with higher $accent(w, "^")_i$.

== Nerfacto <nerfacto>
After the publication of NeRF 2020, many other researches on this method have been published to improve or extend the original NeRF method. One of them is Nerfacto @tancik_nerfstudio_2023, which takes advantage of several NeRF variants. While vanilla NeRF works well when images observe the scene from a constant distance, NeRF drawings in less staged situations show notable artifacts. Due this problem Mip-NeRF @barron_mip-nerf_2021 addresses this by adopting concepts from mip mapping and enabling a scale-aware representation of the radiation field. By introducing integrated position coding, areas in space can be coded more efficiently, leading to better rendering results and more efficient processing. This helps with generating images from different distances which is more realistic for capturing images. To reduce the time for training, Nerfacto use the Insant-NGP's method @muller_instant_2022. Instead of feeding the input coordinates into the network directly (as with NeRF), the coordinates are processed using multi-resolution hash coding. Since the hash coding already captures many details, the actual MLP can be much smaller than with NeRF. The MLP is mainly used to process the features provided by the hash coding and to calculate the final values (color and density). Summarized, Nerfacto is faster and more accurate than NeRF and as the integrated NeRF variant from Nerfstudio, which I used for the implementation, it is supported and worked on.

== LiDAR Simulation <lidar-simulation>
According to @zhang_nerf-lidar_2024, one common approach to LiDAR simulation involves creating a 3D virtual environment and rendering point clouds using physics-based simulations. This method allows for the generation of synthetic LiDAR data. However, Zhang et al. note that these virtual datasets often exhibit significant domain gaps when compared to real-world data, especially when used to train deep neural networks. The primary reason is that virtual environments cannot fully replicate the complexity and intricate details of the real world.

By using a NeRF, the physical processes of the LiDAR rays are not simulated in detail, but the underlying scene is approximated by a neural network. The universal approximation (a feedforward network with a hidden layer and a suitable activation function can approximate any mathematically defined function with arbitrary accuracy under ideal conditions @hornik_multilayer_1989 @hanin_approximating_2017) property of neural networks enables the NeRF to learn the density and radiation information of a real scene so accurately that it enables a precise representation of the scene.

The neural network approximates the spatial distribution of objects and surfaces in a scene and can then deduce which rays are reflected from which points and the resulting distances. Unlike traditional simulations, this approach is not based on explicit physical models, but on a data-driven reconstruction that adapts directly to the observed reality. This reduces the need to fully model the complex physical and optical interactions in the real world.

== Black Box <black-box>
As NeRF is a feed-forward artificial neural network (ANN), it suffers from the disadvantage of being a black box. Interpreting how the model (a trained ANN that is used to approximate a function of interest) uses the input variables to predict the target is complicated by the complexity of ANNs @la_rocca_opening_2022. Even those who create them cannot understand how the variables are combined to produce predictions. Black box prediction models, even with a list of input variables, can be such complex functions of the variables that no human can understand how the variables work together to produce a final prediction @rudin_why_2019. To interpret NeRFs computation, which are only complex functions, as a coordinates, it is not possible to efficiently and quickly define the values while the ANN is computing. It limits the ability to understand how exactly the calculations are done and limits the understanding of the coordinates to the result. One challenge in this implementation is determining the position of a point on an obstacle, which is not feasible for exact values. To address this issue, it is essential to comprehend the underlying computation of NeRF densities. Given that NeRF is a black box, it is only possible to analyze the results, not the computation itself. This restricts the possibility to understand or manipulate for the approximation problem described in ----------------------------------.

== Similar Work
This section will show other studies that are working on the challenge to synthesize LiDAR sensors within a NeRF scene, and what are the uniqueness of my work.

=== NeRF-LiDAR <nerf-lidar>
#figure(image("images/nerf_lidar.png", width: 70%), caption: [The test scene in NVIDIA Omniverse (A) and the resulting point cloud as plot (B) from this application (vertical FOV: 360°, angular resolution: 0.5; horizontal FOV: 100°, angular resolution: 1.8). NeRF-LiDAR from Zhan et. al. (C) and a comparison with a real LiDAR (D).])

#cite(<zhang_nerf-lidar_2024>, form: "prose") also generate LiDAR data by employing real LiDAR data to compute precise 3D geometry as a supervisory signal to facilitate the generation of more realistic point clouds and achieve good results. In contrast to the approach taken by Zhang et al., this thesis synthesizes LiDAR data without the use of additional input data for NeRF. The use of LiDAR data for training offers the advantage of improved results and a more accurate understanding of the scene under consideration. However, this approach does not fully capture the nuances of LiDAR-specific properties, such as ray drop, luminescence, or the permeability of the medium through which the light propagates. While Zhang's research offers valuable insights into specific scenes, my approach may not be as accurate but is more accessible and can be applied to any scene that a NeRF can generate..

== NeRF-LOAM: Neural Implicit Representation for Large-Scale
@deng_nerf-loam_2023 #text(fill: red, [Katographieren mit LiDAR daten und einem NeRF für pose und voxel])

=== A Probabilistic Formulation of LiDAR Mapping with Neural Radiance Fields
@mcdermott_probabilistic_2024 #text(fill: red, [Folgt])

=== SHINE-Mapping
@zhong_shine-mapping_2023 #text(fill: red, [Folgt])

=== PIN-SLAM
@pan_pin-slam_2024 #text(fill: red, [Folgt])
#pagebreak()

= Tools <tools>
This chapter explains Nerfstudio and NVIDIA Omniverse and why I used them for my thesis.

== Nerfstudio <nerfstudio>
Nerfstudio is a PyTorch framework. It contains plug-and-play components for implementing NeRFs. It uses tools for real-time visualization and introduces how to work with and implement NeRFs @tancik_nerfstudio_2023. Nerfstudio is used for using and editing Nerfacto and rendering the scene on the front-end. While Nerfstudio's common use is to implement new NeRFs, this work use a modified version of Nerfacto.

== NVIDIA Omniverse <nvidia-omniverse>
Since the computation from distance in a NeRF depends on the camera parameters, it is easier to use graphical software where the camera values and position can be set and extracted in a virtual environment. For NeRF, it is not necessary to use real images, but for the comparison with a real LiDAR, software is needed that can simulate a LiDAR sensor. One tool that embeds and LiDAR sensors, such as the Sick PicoScan 150, is NVIDIA Omniverse. \
NVIDIA Omniverse is powered by NVIDIA RTX and is a modular development platform of APIs and micro services for creating 3D applications, services and to break down data silos, link teams in real time, and produce physically realistic world-scale simulations with tools like Isaac Sim @nvidia_nvidia_nodate.
#pagebreak()

= Implementation <implementation>
The objective of this project is to establish an origin within a NeRF scene representation and utilize this origin for distance measurement. To achieve this, the implementation requires a frontend for setting the origin and resolutions, visualizing the scene, and parsing the data within the scene representation. Nerfstudio is employed for this implementation (see @nerfstudio).

#figure(image("images/implementation/first_shadow.png", width: 100%), caption: [Activity diagram for the implementation (illustration by author).])<act_dig1>

The principal process is illustrated in @act_dig1. Once the model has been trained, the Nerfstudio 3D visualization (Viser) with LiDAR environment can be initiated with the CLI command #strong[#emph[ns-view-lidar]] and standard Nerfstudio flags. This action opens the frontend, which allows the user to add and position a frustum within the scene. Upon clicking the button in the frontend, a Nerfstudio camera object is generated, and a resolution is established in the backend. This process entails the computation of the specified densities and their subsequent transmission to the frontend.

== Viser <viser>
Viser is a three-dimensional web visualization library utilized within the Nerfstudio framework. The library is utilized for the rendering of both the NeRF scene and the graphical user interface (GUI). The Viser library was utilized to develop the frontend user interface (UI), the point clouds within the scene, and the frustum for the pose. However, Viser is constrained in its functionality, which presents challenges or even impossibilities when attempting to create user interfaces that adhere to current standards for user experience, personas, and user-centered design. While Nerfstudio utilizes the right-handed frame convention (OpenGL), Viser employs left-handed (OpenCV).

== Frontend <frontend>
The frontend is the user interaction area where a frustum with individual color and resolution can be created and positioned to generate points clouds for distance measuring.
#figure(image("images/implementation/frontend.png", width: 70%), caption: [Possible user actions in the frontend. Current scale factor for distance which represents how distances in the scene are scaled before or after calibration with a reference object in the scene (a). Point cloud settings for individual color or size of each point cloud (b). For synthetic Sick LiDAR sensors, the vertical and horizontal angular resolution can be set. So it does not depend on hardware characteristics (c). Ability to use a reference object within the scene to calibrate the NeRF distance (d) (illustration by author) #text(fill:blue, [UI im Bild nicht aktuell. Auch eine maximale distanz kann inwischen gewählt werden]).]) <fig:frontend>

In order to obtain a point cloud within the scene or as a plot, it is necessary for the user to position the frustum. Subsequently, the user may select either a pre configured Sick LiDAR sensor or an individual point cloud. Individual point clouds afford greater flexibility with regard to angular resolution and the quantity of rays. The frontend enables the measurement of two points in the scene, where the frustum is employed to generate the points and to calibrate the scale factor for distance measurement. This may be achieved by utilizing a reference object in the scene with a known size to calibrate the scale factor.

#figure(image("images/implementation/second.png", width: 300pt), caption: [Frontend implementation as activity diagram. After receiving the frustum data from the frontend, the user can choose between Sick LiDAR, individual measurement and calibration. This returns the LiDAR resolution and activates the process to obtain the densities in the backend (illustration by author).])

== Backend <backend>
- Kurze Beschreibung, wie das Backend verläuft. Nur, dass eine Kamera generiert wird. Diese dann durch verschieden Schritte das ANN anspricht und anschließend eine Liste mit Daten (Density, positions) zurück gibt.
- Die Änderungen die ich hinzugefügt habe, z.B: das hinzufügen eines eigenen Kamertyps, da Nerfstudio auf echte Kameras ausgelegt ist, und nicht auf LiDAR mit 360 Grad.

== Scene recognition
NeRF as an artificial neural network learns from differences within images. These differences can be in terms of color, texture, shape, or lighting variations. When a scene contains homogeneous areas—regions with uniform color and texture—NeRF faces challenges in accurately reconstructing these areas spatially.\

In the Save-Robotic scenario from Sick, this issue was observed with the floor. Although NeRF was able to render the floor correctly from a perspective view, the spatial reconstruction was not precise enough for the intended distance measurements. This problem arises from two main factors:\


- #strong([Homogeneity:]) NeRF learns to model a scene by mapping spatial coordinates $(x,y,z)$ and viewing directions $(θ,ϕ)$ to color values $(R G B)$ and density $(σ)$. It does this by minimizing the loss function, which represents the difference between the rendered outputs and the ground-truth images during training. \

  In homogeneous areas like a uniformly colored floor, there is a lack of distinguishable features or variations that can help the network associate different spatial positions with specific outputs. This absence of variation means that while NeRF can reproduce the appearance of the floor from known viewpoints, it may not accurately capture the spatial depth or geometry in these regions. Consequently, certain spatial measurements may not achieve the desired level of accuracy.

- #strong([Reflection:]) Reflective surfaces introduce additional complexity because they cause view-dependent appearance changes. In the case of a homogeneous and reflective floor, reflections can cause the floor to appear differently from various perspectives. While NeRF is designed to model view-dependent effects, the reflections on a homogeneous floor do not provide consistent spatial cues that correlate with actual geometric differences.\

  This inconsistency can affect the network's ability to learn an accurate spatial representation of the floor. The reflections introduce variations that are not indicative of spatial changes, potentially leading to inaccuracies in the reconstructed spatial properties. 

#figure(image("images/implementation/floor/floor_illustration.png", width: 70%), caption: [Expected ground rendering (A). Every pixel in the rendered scene is perspective. The spatial view depends on the environment, which is also perspective. The ground is scattered in different distances from the viewpoint. This makes it difficult to measure similar distances to the original scene (B,  illustration by author).]) <fig:floor> 

As mentioned is the perspective view as expected. Due the issue to recognize the correct location of the floor resp. the pixel of the floor, the distances from the pixels are not correct. As shown in @fig:floor the pixels are scattered. \

The following illustrations depict various floor plans. The leftmost column presents an image from the scene in which point clouds are generated from an origin using the distance measurement algorithm that I have used. It is evident that in the first two examples, the floor is not recognized, except for the shadow of the box. Points outside the shadows are mostly located below the floor level, which is more clearly illustrated in the middle image, a plot created for better visualization. The varied colors of the points indicate different distances for each point. When the colors of points within the shadow are similar to the surrounding colors, the floor is detected. The last column displays a graph representing multiple tests conducted in each scene. For these tests, 160 samples with 50 rays at different positions were used, as illustrated in @fig:floor_ex below.

#figure(image("images/implementation/floor/graph_example.png", width: 80%), caption: [Left-handed: The illustration for the floor test shown in the graphs below. Each scene has 50 samples with 160 rays and different position. The graphs below represent the average of each 160 rays from each sample. The origin of each sample is the same. Therefore, the distance in the center is smaller than in the edge. A good result is illustrated in the right image. where the vertical axis represents the 160 rays and the horizontal axis represents the distance.]) <fig:floor_ex>

#let width = 120pt
#let height = 100pt

#let graph_width = 190pt
#figure(
  align(
    grid(
      columns: 3,
      column-gutter: 3mm,
      align: bottom,
      image("images/implementation/floor/grey1.png", width: width),
      image("images/implementation/floor/grey2.png", width: width),
      image("images/implementation/floor/grey_graph.png", width: graph_width),
    ),
  ),
  caption: [Left: Point cloud of the NeRF scene with gray ground with reflections. The points are only on the cube and the ground. Middle: Point cloud plotted for better recognition. Right: Plot described below without recognizable pattern. To illustrate the problems of spatial recognition by ANN from NeRFs by using a homogenous floor and reflections]
)
  
The first example has a homogeneous gray floor with reflections. The left image shows some artifacts at the bottom of the image. This is due to the reflections on the original scene. The point cloud on the cube and the shadow on the floor are well visible. There are more points outside the shadow that are hard to see. These points are below the ground level. This is better visible on the middle image which is the plot from the same position. The graph is difficult to evaluate. Most of the areas below the frustum have no values because the distance is limited to 50 meters for better representation. The other parts depend on the neural network. There is no recognizable pattern. . There is no recognizable pattern. It is important to note that the point cloud from the left image and the middle image are similar but not the same. Both point clouds have the same origin in the same scene, but the results are not the same because of the use of a neural network to approximate the densities. The results of the graph use the same scene but not the same positions as the first two images. Therefore, they should not be compared directly.

#figure(
  align(
    grid(
      columns: 3,
      column-gutter: 3mm,
      align: bottom,
      image("images/implementation/floor/grey_no_reflection.png", width: width),
      image("images/implementation/floor/grey_no_reflection2.png", width: width),
      image("images/implementation/floor/grey_no_reflections_graph.png", width: graph_width),
    ),
  ),
  caption: [Left: Point cloud of the NeRF scene with gray ground without reflections. The points are only on the cube and the ground. Middle: Point cloud plotted for better recognition. Right: Plot described below without recognizable pattern. To illustrate the problems of spatial recognition by ANN from NeRFs by using a homogenous floor and reflections]
)

The second example also has a homogeneous gray floor, but without reflections. There are no artifacts on the floor, which also happens in other test scenes where the reflections are removed. As in the first example, most of the points are on the cube and the shadow. As the second image shows, there are also a few points outside these areas. We can see a kind of gradient under the frustum, which has more points than the first example. The graph also has some parts without values. The high peak could be due to fewer images from the training (which is better to see in the fourth example). The images are focused on the cube, the shadow, and the ground around it. Some of the samples are out of focus. The right side of the graph still seems chaotic, but less random than the first example with a constant average distance of the points. This is better but not a good result because the edges shout have higher distances.

#figure(
  align(
   grid(
    columns: 3,
    column-gutter: 3mm,
    align: bottom,
    image("images/implementation/floor/checker.png", width: width),
    image("images/implementation/floor/checker2.png", width: width),
    image("images/implementation/floor/checker_graph.png", width: graph_width),
  )
),
  ),
  caption: [Left: Point cloud of the NeRF scene with gray ground with reflections. The points are only on the cube and the ground. Middle: Point cloud plotted for better recognition. Right: Plot described below without recognizable pattern. To illustrate the problems of spatial recognition by ANN from NeRFs by using a homogenous floor and reflections]
)

The third example use a checkerboard pattern with different colors. The moste points are on the floor and not only on the shadow and the cube. The use of this pattern led to a significantly better result as the examples before, even if the checkerboard pattern is repetitive. The graphs shows in the middle an approximation to an good and recognizable pattern in average. 

#align(
   grid(
    columns: 3,
    column-gutter: 3mm,
    align: bottom,
    image("images/implementation/floor/interference.png", width: width),
    image("images/implementation/floor/interference2.png", width: width),
    image("images/implementation/floor/interference_graph.png", width: graph_width),
  )
)

The last virtual example is the same pattern from the test scene I used. This scene also uses a checkerboard pattern, but with some interferences in between. This makes most of the areas on the floor unique. The first two images are similar to the previous example. On closer inspection, the previous example seems to be better. Fewer outlines. The graph, on the other hand, shows a much better result than the previous example. As mentioned, the left side of the graph has more peaks than the right side, which is more in the focus of the images from the training.

#align(
   grid(
    columns: 3,
    column-gutter: 3mm,
    align: bottom,
     image("images/implementation/floor/center3.png", width: width),
     image("images/implementation/floor/center.png", width: width),
   image("images/implementation/floor/center2.png", width: width),
  )
)

A direct comparison from each floor and from different angles. The difference between the gray pattern with and without reflection and the checkerboard with and without interference is not visible in these images. But these differences are important for accurate measurement.

#align(
   grid(
    columns: 3,
    column-gutter: 3mm,
    align: bottom,
    image("images/implementation/floor/red.png", width: width),
    image("images/implementation/floor/red2.png", width: width),
  )
)

A 3D scene representation from a real scene with my self-made reference object on a red carpet. Even with different pattern in the carpet, only this parts on the floor are recognized which are affected from a shadow. Which means, that this issue also affects on floors which has minimal differences like signs of use on a industrial floor. 

== User interactions
This section presents a comprehensive overview of the user's available interactions.
Due to the constraints of limited resources and render time, two distinct types of point clouds have been implemented. A non-clickable point cloud is provided for visual demonstration purposes only. In contrast, the clickable point cloud allows users to click on each point, and the distance from the origin to that point is displayed. In the non-clickable method, the point cloud itself is an object in the scene. In contrast, in the clickable point cloud, each point is a single object in the scene. This approach is considerably more resource-intensive and time-consuming than the first method.

=== Synthesize Sick LiDAR
To illustrate the methodology, two LiDAR demonstrations are presented, utilizing the Sick LiDAR sensor (PicoScan 100 and MultiScan 100). While the angular resolution of this sensor is fixed, the NeRF implementation is more dynamic. The original resolution is displayed in the interface, but the user may select a differential option. This is particularly relevant for the MultiScan, which has a vertical angular resolution of 0.125 for 360°. The computation of these rays is a time-consuming process. The settings for each LiDAR demonstration are saved in a JSON file, and each demonstration is loaded dynamically in the front end. Additional demonstrations and capabilities can be incorporated in future implementations.

=== Individual measuring
While the demonstration of LiDAR sensors has defined angular and resolution parameters, users may also employ the individual measurement. This method allows for the independent adjustment of vertical and horizontal resolution, as well as the number of rays. This functionality enables the comprehensive scanning of an entire scene from a single origin or a single ray.

=== Precise measuring
This implementation allows the user to make precise measurements between two points within the scene. To do this, the user is able to select a clickable point cloud in the same way as other individual point clouds. After selecting a point, the user can then add this coordinate as a measurement point (see @fig:frontend). Once a second measure point has been added, the distance between the two points is displayed. Note that since a NeRF scene is a virtual scene, this functionality also works through walls and objects.

=== Calibration
If the camera parameter are unknown, it is necessary to calibrate the scale factor within the scene. For this, the precise measurement includes a possibility for calibrate this. For the calibration, an object in the scene is needed where the size of this object are known. In best case, this object is predestined for measurement with colors and patters and forms which the ANN can recognize easily.

#figure(image("images/implementation/known_object.png", width: 200pt), caption: [Self-made reference object with a defined size and different patters for the ANN to recognize the object within the scene (image by author).])

To use this calibration, the user can use the two measuring points (see Measuring Points). Then click on "Open Calibrate Modal" (see @fig:frontend), which opens a modal in which the distance from the reference object (the two points) in relation to the correct size of the object can be used to calibrate the scale factor. After this calibration, every measurement will be calculated with this scale factor. This can be easily undone.
#pagebreak()
= Density Distance Algorithm <density-distance-algorithm>
A LiDAR sensor emits lasers in one direction and calculates the time it takes for the light to travel to an obstacle, reflect and return to the sensor. Because the speed of light is constant, this measurement is very accurate.
A NeRF Scene does not know any obstacle or coordinates. After casting a ray, it estimates the density volume along a ray based on Beer-Lambert-Law, which provides a description of the effects arising resulting from the interaction between light and matter @mayerhofer_bouguerbeerlambert_2020: 

== Single ray <single-ray>
It is important to understand the functionality of a single ray. Each ray returns a list of density values and their corresponding coordinates. Each value is computed based on the Beer-Lambert method described above. \
NeRF can estimate new perspectives by learning a continuous and differentiable function of the scene that allows views from arbitrary angles. It effectively interpolates between the known perspectives using the learned 3D properties of the object. \
Due to the fact that each ray has infinite points along the ray, a sampling rate is needed (see @volume_sampling). If the estimated density value along the ray does not increase, a sampling point is set on the ray after a certain interval. When the density value increases, indicating that the ray has hit an object, both the density and the sampling rate increase. After the ray passes through the object, the sampling interval becomes larger again, as there is less relevant information behind the obstacle. \
For testing, a test scene is created with NVIDIA Omniverse. This test scene is built with a $1³$ m pattern to easily estimate distances and a color pattern for the ANN to better recognize the scene. To analyze the behavior of a single ray and its density values, several single rays are simulated in this scene with an exact distance of 1m to a wall within the scene.

#figure(image("images/Density_Distance_Algorithm/dda1.png", width: 100%, ), caption: [Sampling points along a ray. Constant sampling rate before collision with the object (a). High sampling rate due to the wall (b) and lower sampling rate with increasing distance (c).  For better illustration, the sample points are larger in C (illustration by author).])

=== Single ray pattern
#text(fill: blue, [Hier möchte ich noch auf das Pattern eingehen, also wie sich die Density verhält, wenn eine Kollision stattfindet. Dazu habe ich über 20000 Plots erstellt, von denen ich noch nicht genau weiß wie ich sie einlesen soll, da ich keine Rohdaten dafür habe. Als Beispiel sind hier vier Bilder, die den gleichen Ray und seine Density über Zeit zeigt. Das erste Bild (oben links) ist der komplette Ray und die weiteren immer etwas näher an einem roten Punkt, der genau 1 Meter zeigt (also die Position, die am besten wäre). Der rote Punkt ist schwierig zu sehen, man muss näher heranzoomen. Es ist zu sehen, dass der Peak der Density nicht ausschlaggebend für die beste Entfernung ist, was die Notwendigkeit, aber auch die Komplexität, den optimalen Punkt zu finden, unterstreicht. Von jedem der 511 Rays habe 40 Plots.])

#table(
  columns: 2,
  stroke: none,
  image("images/Density_Distance_Algorithm/single_ray/ray_id_108.png", width: 120%),
  image("images/Density_Distance_Algorithm/single_ray/ray_id_108_range_28.png", width: 120%),
  image("images/Density_Distance_Algorithm/single_ray/ray_id_108_range_15.png", width: 120%),
  image("images/Density_Distance_Algorithm/single_ray/ray_id_108_range_3.png", width: 120%)
)

=== Accuracy <accuracy>
An important question is whether it is possible to simulate the distance as accurately as a LiDAR sensor. To test this, several single rays are cast at a distance of 1m to the obstacle and the density value closest to 1m are plotted. For 11 different locations with different colors where the ray hits the wall, 50 rays are cast. The total average distance of all 550 rays are 1.000014739m which is a total deviation from approximately 15μm. The average density value on this exact point are 86.447. The interesting part is the difference between the different locations:
<tab:my_table>

#align(
  figure(
    table(
        columns: 3,
        stroke: none,
        inset: 6pt,
        [#strong[Ray]],
        [#strong[Average Distance in m]],
        [#strong[Average Density Value]],
        [1], [0.999911142], [20.3412],
        [2], [0.999950498], [58.9121],
        [3], [0.999995336], [59.6347],
        [4], [0.999994195], [#highlight(fill: rgb(122, 205, 255), "389.1807")],
        [5], [1.000088757], [194.9974],
        [6], [1.000124663], [#highlight(fill: rgb(255, 174, 122), "1.6392")],
        [7], [1.000016545], [5.7899],
        [8], [1.00001], [11.2547],
        [9], [1.000156424], [102.6740],
        [10], [0.999937118], [96.3068],
        [11], [1.000124663], [1.63928],
      ),
  )) <tab:my_table>

#let width = 38pt
#let height = 40pt
#grid(
  columns: 11,     // 2 means 2 auto-sized columns
  rows: 2,
  gutter: 2mm,    // space between columns
  image("images/Density_Distance_Algorithm/single_ray_table/1.png", width: width, height: height),
  image("images/Density_Distance_Algorithm/single_ray_table/2.png", width: width, height: height),
  image("images/Density_Distance_Algorithm/single_ray_table/3.png", width: width, height: height),
  image("images/Density_Distance_Algorithm/single_ray_table/4.png", width: width, height: height),
  image("images/Density_Distance_Algorithm/single_ray_table/5.png", width: width, height: height),
  image("images/Density_Distance_Algorithm/single_ray_table/6.png", width: width, height: height),
  image("images/Density_Distance_Algorithm/single_ray_table/7.png", width: width, height: height),
  image("images/Density_Distance_Algorithm/single_ray_table/8.png", width: width, height: height),
  image("images/Density_Distance_Algorithm/single_ray_table/9.png", width: width, height: height),
  image("images/Density_Distance_Algorithm/single_ray_table/10.png", width: width, height: height),
  image("images/Density_Distance_Algorithm/single_ray_table/11.png", width: width, height: height),
  [Ray 1], [Ray 2], [Ray 3],[Ray 4],[Ray 5],[Ray 5],[Ray 7],[Ray 8],[Ray 9], [Ray 10], [Ray 11]
)
it should be mentioned, that the color is not directly responsible for the density value. It is the accuracy itself, because the density value increases greatly on the positions where the ray is in front of, on, or inside an obstacle. That the object was sometimes recognized better and sometimes worse.\
#text(fill: blue, [Das ist vermutlich zu ungenau beschrieben und kaum belegbar. Die Farbe ist natürlich wichtig für das Erkennen der Szene. Was ich meinte, ist, dass die die Gründe eher darin liegen wie gut die Szene erkannt wurde. Das sollte ich nochmal überarbeiten.])

== Methods

The employment of multiple lasers facilitates the generation of a point cloud representation of the surrounding environment. These sensors are specifically designed for real-time applications. However, the utilization of a NeRF to synthesize data in real-time remains a challenge. The computation of distance between two points within a NeRF scene an generating a point cloud necessitates the completion of several steps.
#v(2mm) 
#strong[Determine the points:] This represents a significant challenge with regard to implementation. In other graphical software, such as Unity, NVIDIA Omniverse, and Blender, the coordinates of each edge, vertex, or face are explicitly defined. However, in NeRF, these coordinates are not known. To illustrate this concept in a simplified manner, each scene in NeRF can be conceptualized as a "memory" of the neural network, which learns to generate RGB and density values from a multitude of perspectives and positions by minimizing the discrepancy between the original image and its internal representation. This is referred to as the loss function. This discrepancy in scale and coordinates between the original scene and the NeRF scene presents a challenge. Although this implementation assumed that the local origin in the global scene could be set manually with a visible object within the scene that is movable in runtime, the primary issue lies in determining the appropriate distance between this local origin and each potential obstacle that a ray might encounter. As in @volume_sampling described, the density value increases close to an obstacle. To address this issue, multiple methods are developed and tested to estimate the sample point closest to the obstacle by recognize the increasing density value.
#v(2mm)
#strong[Casting rays:] Once the origin within the scene has been established, rays must be cast in order to obtain the requisite distance measurement and point cloud reflection. The rays should be dependent on the local origin and not on the scene perspective, which are typically employed in NeRF for scene representation.
#v(2mm)
#strong[Distance measurement:] To measure the distance between two points, the Pythagorean theorem is applied. A key challenge, however, lies in the fact that the scale within the scene does not correspond accurately to that of the original scene, particularly when camera parameters are estimated rather than precisely known during training.
#v(2mm)
#figure(image("images/related_work/lidar_vs_nerf.png"), caption: [Simplified illustration depicts the concept of LiDAR and the rationale behind this thesis. While a time-of-flight sensor allows light to pass through and measures the time it takes for light to travel to an obstacle and reflect back to a receiver (illustrated on the left), a NeRF does not actually cast rays within the scene. The challenge is to identify the sample point that is closest to the obstacle and compute the distance between the origin and the sample point, as the coordinates of the origin and the sample point are within the scene.])

As illustrated in @fig:impicit it is challenging to obtain the requisite coordination to measure the distance between the two points. While the distance can be readily calculated in NVIDIA Omniverse due to the availability of known coordinates, the coordinates in a NeRF scene are dependent on the original images and their associated parameters. It is feasible to approximate the same coordinates as in NVIDIA Omniverse. However, due to the issue of infinite points in space and the lack of knowledge regarding the coordinate of each object within the scene, it is not possible to obtain the exact coordinates, which represents a significant challenge in the implementation of this approach with LiDAR sensors, which are highly precise. Nevertheless, even when it would be feasible to obtain the exact positions, the distance would not be accurate, given that the NeRF scene lacks any references regarding scale. This understanding also depends on the camera parameters. Another Problem is to obtain the closest point on an obstacle. A NeRF approximates the original scene by approximates the RGB and density values. These values are not exact representations of the scene. In the real world or in graphical software, it is possible to use a pen or click on an object because its coordinates are known and defined. However, in a NeRF, every pixel is only an approximation of an RGB value, influenced by the density value depending from images. This means that it is not possible to take a pen and mark a specific point, because the exact position and properties of that point are not explicitly defined in the model. \

#figure(image("images/introduction/implicite.png", width: 100%), caption: [Comparison between implicit and explicit coordinates in NVIDIA Omniverse and a NeRF scene representation. The use of NVIDIA Omniverse to calculate the distance between two known points in a 3D scene using the Pythagorean theorem (A). The coordinates as user input: $x'_a , y'_a , z'_a$ for the neural network $f_theta$ which interprets these coordinates by computing pixels in the space it learns from training (B). Illustration of the problem of getting a point closest to an obstacle in NeRF (C).#text(fill:blue, [Ich suche immer noch nach einer besseren Darstellung für das rechte Bild. Es wird nicht ganz klar, was das Problem darstellt.])])<fig:impicit>

=== Not used methods <methods>
#text(fill:blue, [Alle Abschnitte dieses Kapitels müssen noch einmal überarbeitet und kontrolliert werden. Vor Allem das "In this method" ist sehr nervig.])\
This section shows the methods for estimating the sample point closest to an obstacle. For this purpose, all rays thrown are analyzed and several methods are tested. The rays themselves are not modified.

To simulate a NeRF scene, an origin in the scene is required, which can be added as a frustum within the scene. Depending on user input, this frustum simulates a ray within the scene and obtains the density values along a ray.

The core idea is to estimate the density sample point $sigma$ at the position $x$ closest to the obstacle where the ray intersects, and then calculate the distance between the origin and this estimated point. Several methods are used and tested for this estimation.

==== Threshold method <simple-threshold-method>
For each ray, I sample the volume densities $sigma(x)$ at discrete points along the ray path. By defining a threshold $sigma_(t h)$, which determines the density above which a relevant interaction is assumed, the algorithm searches for the first position $x_(t h)$ where $sigma(x_(t j)) gt.eq sigma_(t h)$. This corresponding position $x_(t h)$ is then used to determine the distance to the object. \

While this method is commonly used for testing, it has two problems. First, the use of a threshold itself. Even though the use of a threshold is necessary in any method, it is a goal of this work that the user can use this implementation without defining a threshold because of the second problem: the definition of the threshold. As I mentioned before, the density value closest to the obstacle is arbitrary and different from ray to ray and scene to scene. Even a ray at the same position will give different values. While the test results are good with known distance, it requires the distance to calibrate $sigma_(t h)$, which makes it useless.

==== Difference method
his method focuses on the increases in density along a ray. If $abs(Delta sigma) gt.eq Delta_(t h) $, where $Delta sigma = sigma_i minus sigma_(i-1)$ and $Delta_(t h)$ represents a threshold, then $sigma_i$ corresponding position $x_i$ is used for the distance calculation.

This method is similar to the first one, but the threshold should not be set by the user. As demonstrated by @single-ray, the density value of the required density closest to the obstacle will vary from ray to ray, even if the origin and direction of the ray are the same. While the first method sets an absolute threshold that is independent of each ray, the focus on increasing each density makes each ray independent of other rays. Since the densities are too different, this method suffers from the same problem as the first method, but with better results and usable without threshold calibration.

==== Z-Score method
In this method, I use the z-score to standardize the density values along each ray and identify points where the density deviates significantly from the mean. For each ray, I calculate the mean $mu$ and the standard deviation $s$ of the density values ${sigma_i}$. The z-score for each density value is calculated as:
$z_i = (sigma_i - mu) / s$

==== Standardized method
In this method, I standardize the volume densities along each ray to account for global variations. First, I compute the global mean $μ$ and the global standard deviation $s$ of all volume densities across all rays. For each ray, I standardize the density values $σ_i$ along the ray using $accent(σ, ~)_i=(σ_i -μ)/s$.

Then I defined a threshold $Delta_(t h)$ that determines when a density is considered significant. Search the standardized density values along the ray and identify the first point $x_(i∗)$ at which $accent(σ, ~)_(i∗) gt.eq Δ_(t h)$ holds. This position is used to calculate the distance to the object.

==== Accumulate method
In this method, the volume densities $σ_i$ accumulate along each ray until the cumulative density reaches or exceeds a predefined threshold $Sigma_(t h)$. For each sample point $x_i$ along the ray, the cumulative density $S_i =Sigma_(k=1)^i σ_k$ is calculated. The first point $x_(i∗)$ where $S_i gt.eq Sigma_(t h)$ is found. This position is used to calculate the distance to the object.

=== Maximum density change
This method calculates the absolute density changes between successive sample points along each ray. Let $σ_i$ be the volume density at point $x_i$. The density change between points $x_(i-1)$ and $x_i$ is calculated as $Delta_(sigma_i) = abs(sigma_i -sigma_i -1)$. The index $i^∗$ at which the density change $Delta_(sigma_i)$ is maximized is determined: $i^∗ = arg max Delta_(σ_i)$. The corresponding position $x_(i^∗)$ is assumed to be the position where the ray hits the object, and the distance is calculated.

==== Average density change
Compared to the previous method, where I identified the point along the ray with the maximum density change between consecutive samples, this method similarly calculates the absolute density differences between consecutive points along each ray. However, instead of relying solely on the maximum density change, I calculate the average density difference $accent(Delta_sigma, -)$ over the entire ray. I then define a dynamic threshold $Delta_(t h) = k * accent(Delta_sigma, -)$, where $k$ is a scaling factor.

=== Used method
Once the frustum has been positioned with respect to the origin and direction, the vertical and horizontal angular resolution has been obtained, and the button has been pressed to generate the point cloud within the scene, a Nerfstudio camera object will be created. Furthermore, all densities and locations along each ray will be saved in a list.
It should be noted that transmittance along a ray is defined in Nerfstudio as:
$ T(t) = exp(- integral_(t_n)^t sigma (r (s)) d s) $ Where $sigma (r (s))$ is the volume density along the ray depending on the position $r (s)$ on ray $r (t) = o + t d$.
Since the ray is scanned at discrete points, the integral can be approximated by a sum over the segments:\
#align(center,$T_n = product^n_(i=1) exp(-sigma_i ⋅ Delta_(s_i))$) where:\ 
- $sigma_i$ is the density on point $i$ \
- $Delta_(s_i)$ is the distance between point $i-1$ and $i$ \
- $T_o = 1$ the initial transmittance \

A collision is detected when transmittance $T_n$ falls below the threshold: $T_n gt.small T_(t h)$, where $T_(t h) = 10^(-200)$.

#pseudocode-list()[
  + *Function* find_collision_with_transmittance(ray_locations, ray_densities, threshold):
    + Set total_distance to 0
    + Set transmittance to 1.0  // Initial transmittance value
    + origin = ray_locations[0]

    + *For* each location, density in ray_locations[1:], ray_densities[1:]:
      + calculate distance between origin and current location
      + add this distance to total_distance

      + calculate delta_transmittance as exp(-density \* distance)
      + update transmittance by multiplying with delta_transmittance

      + *If* transmittance < threshold:
        + Return: total_distance, current location, current density  // Collision found
      + *End If*
    + *End For*
    + Return: No collision found  // If no collision is detected
  + *End*
  ]

#pagebreak()

= Results
In this chapter I will present some test with my test scene and a real scene with a refence object. It was not possible to test it with a real LiDAR.

== Test scene
To gain a more comprehensive understanding of this implementation, a trial was conducted in a test environment. The test scene was trained with known extrinsic and intrinsic parameters and with interference patterns for enhanced recognition due to the training. Given the known position, a distance of one meter in the original NVIDIA Omniverse scene is equivalent to one meter in the test scene. This facilitates the measurement of distance and composition with the original scene.

In order to test the hypothesis, 2560 rays were cast onto different surfaces at varying distances. At each position, 10 rays were cast, and the distance was increased by 0.5 meters up to a maximum of 4.5 meters, which represents the border of the scene. Thereafter, the position was changed to another surface.

#figure(
  table(
      columns: 2,
      stroke: none,
      align: start,
      [Average Deviation:], [0.0001116 m],
      [Average Density:], [233.069]
  ),
)


#set table(
  stroke: none,
  gutter: 0.2em,
  fill: (x, y) =>
    if x == 0 or y == 0 { gray },
  inset: (right: 1.5em),
)

#show table.cell: it => {
  if it.x == 0 or it.y == 0 {
    set text(white)
    strong(it)
  } else if it.body == [] {
    // Replace empty cells with 'N/A'
    pad(..it.inset)[_N/A_]
  } else {
    it
  }
}

#align(center,
  table(
    columns: 3,
    align: center,
    gutter: 5pt,
    [], [Average Deviation with:], [Average Density with:], 
         [1 meter],[-0.00074724], [245.23731445],
         [1.5 meter],[-0.00193337], [168.95489944],
         [2 meter],[-0.00030199], [156.13135481],
         [2.5 meter],[0.00103657], [130.91514499],
         [3 meter],[0.00167161], [141.78231862],
         [3.5 meter],[0.00099374], [180.60095075],
         [4 meter],[0.00077826], [277.47396092],
         [4.5 meter],[-0.00060454], [563.45454628],
  )
)
#figure(
  image("images/results/average_re.png", ), caption: [Average distance from all samples]
)
#align(
   grid(
    columns: 2,
    row-gutter: 3mm,
    column-gutter: 3mm,
    align: bottom,
    image("images/results/1_fin.png"), image("images/results/1_5_fin.png"),
    [1 meter], [1.5 meter]
  )
)
#align(
   grid(
    columns: 2,
    row-gutter: 3mm,
    column-gutter: 3mm,
    align: bottom,
    image("images/results/2_fin.png"), image("images/results/2_5_fin.png"),
    [2 meter], [2.5 meter]
  )
)
#align(
   grid(
    columns: 2,
    row-gutter: 3mm,
    column-gutter: 3mm,
    align: bottom,
    image("images/results/3_fin.png"), image("images/results/3_5_fin.png"),
    [3 meter], [3.5 meter]
  )
)
#align(
   grid(
    columns: 2,
    row-gutter: 3mm,
    column-gutter: 3mm,
    align: bottom,
    image("images/results/4_fin.png"), image("images/results/4_5_fin.png"),
    [4 meter], [4.5 meter]
  )
)

== Real scene
#text(fill: blue, [Hier würde ich gerne noch Tests in einer realen Szene machen, da ich aber keine reale Umgebung habe, mit der ich Kamerapositionen genau bestimmen kann, könnte dies auch wegfallen.])

== Advantages and Limitations
LiDAR sensors are not perfect. Distance measurements from LiDAR sensors can have deviations from real distances. Up to 20 cm at high angles of incidence @laconte_lidar_2019. Poorly reflecting or highly absorbing materials can affect the backscatter of the laser ray, resulting in inaccurate measurements. One study shows that the range of LiDAR sensors can be reduced by up to 33% under these conditions @loetscher_assessing_2023. Other influencing factors are complex terrain, complexity, measurement height, surface roughness and forest, atmospheric stability and half cone opening angle @klaas-witt_five_2022. An average deviation of 0.1116 mm is a better result than expected at the beginning of this thesis, even in a test environment. Unlike a LiDAR sensor, a NeRF does not need real rays to measure distance within the scene. This makes it possible to measure the distance between two points in the scene, even if there are some obstacles between them, as long as those points are known. To use a NeRF for 3D scene representation, it is easier to reconstruct the scene. 
#v(3mm)
The main limitation of this application is, that it can not used in realtime, which is a big benefit for real sensors. In view of the fact that a neural network has to be trained for a NeRF, it is doubtful that this will be achieved in the next few years, even with better hardware and technology. The second limitation is that factors of a LiDAR sensor are not implemented due to the limited time of this thesis. A LiDAR sensor has many influences on the result, as is showable in this work. Distance measurements are only taken when an emitted LiDAR ray reflects off an object and returns to the sensor. Cases where ray do not return are called ray drop @mcdermott_probabilistic_2024. This ray drop needs physical correct calculation.

...

#text(fill: blue, [Das Kapitell ist noch nicht fertig geschrieben. Alles unterhalb ist nicht relevant für euch.])
#pagebreak()

= Conclusion <conclusion>

#pagebreak()

= Outlook <discussion>
In this chapter I will propose further research about this study.

== Psyhsical Propeties
As mentiont, LiDAR results depends on physical properties. While it es difficult to simulate such properties it would be interesting if such computation can done with one ore multiple neural networks implemented in NeRF.

== Medium of Velocity of Light
The normal use case of an LiDAR-Sensor are in Air where the different between the medium through what the light has pass through are similar to vacuum. The benefit of this applications is, that real rays are not needed ans so on, it can also be used for other mediums, like water or glass.

== Gaussian splat
After 

#pagebreak()

= Discussion <discussion>
#pagebreak()

= Use of AI in this thesis <use-of-ai-in-this-thesis>
The Use of AI has a lot of benefits for research purpose but it is also necessary to understand the rules by using an AI

#table(
  columns: 3,
  stroke: none,
  table.header(
    [*Name of AI*], [*Why used*], [*Where*],
  ),
  [ChatGPT], [For ....], [overall],
)
#pagebreak()

= Declaration on honest academic work <use-of-ai-in-this-thesis>
With this document I, Patrick Kaserer, declare that I have drafted and created the piece of work in hand myself. I declare that I have only used such aids as are permissible and used no other sources or aids than the ones declared. I furthermore assert that any passages used, be that verbatim or paraphrased, have been cited in accordance with current academic citation rules and such passages have been marked accordingly. Additionally, I declare that I have laid open and stated all and any use of any aids such as AI-based chatbots (e.g. ChatGPT), translation (e.g. Deepl), paraphrasing (e.g. Quillbot) or programming (e.g. Github Copilot) devices and have marked any relevant passages accordingly.\
\
I am aware that the use of machine-generated texts is not a guarantee in regard of the quality of their content or the text as a whole. I assert that I used text-generating AI-tools merely as an aid and that the piece of work in hand is, for the most part, the result of my creative input. I am entirely responsible for the use of any machine-generated passages of text I used. 
I also confirm that I have taken note of the document "Satzung der Hochschule Furtwangen (HFU) zur Sicherung guter wissenschaftlicher Praxis" dated October 27, 2022 and that I have followed the statements there.\
\
I am aware that my work may be examined to determine whether any non-permissible aids or plagiarism were used. I also acknowledge that a breach of § 10 or § 11 section 4 and 5 of HFU’s study and examination regulations’ general part may lead to a grade of 5 or «nicht ausreichend»  (not sufficient) for the work in question and / or the exclusion from any further examinations
#place(dy: 150pt, [Place, Date])
#place(dy: 150pt, dx: 300pt,  [Signature])

#pagebreak()

#bibliography("references.bib", full: false, style:"apa")
