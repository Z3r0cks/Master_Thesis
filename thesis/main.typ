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
#strong(text([Master Thesis], size: 37pt, baseline: 20pt))

#text([by], size: 15pt, baseline: 8pt) \

#strong(text([Patrick Kaserer, B.Sc], size: 20pt))

#text([This thesis submitted for the degree of], size: 15pt, baseline: 17pt) \

#strong(text([Master of Science (M.Sc)], size: 20pt, baseline: 10pt))

#text([on the faculty of #strong[digital media]], size: 15pt) \
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
#pagebreak()

= Abstract
#pagebreak()

= Zusammenfassung
#pagebreak()

= Preface
#pagebreak()

= Table of Contents
#outline(depth: 4, indent: 2em, title: none)
#set heading(numbering: "1.")
#set page(numbering: "1")
#counter(page).update(1)

#context counter(heading).update(0)
#set page(footer: auto)

= Introduction <introduction>
#left_sign
Sensors enable machines to perceive and interpret the world around them. By allowing humans, through machines, to observe and measure aspects of the environment beyond human capability, sensors foster the development of technologies and machines that enhance our understanding of the world, physics, human behavior, and more. They are, therefore, indispensable for technological advancement.

This thesis explores whether a synthetic LiDAR sensor can be generated within a Neural Radiance Field (NeRF) framework. By simulating LiDAR sensors in a NeRF, the aim is to determine the feasibility of virtual sensor models, offering insights into which physical sensors may be suitable or necessary for real-world applications. It explores the advantages and disadvantages of this approach.

Industrial applications can be dangerous. The use of inappropriate sensors can cause injuries or worse. Testing potential sensors in a virtual environment, such as a NeRF, is safer and more efficient than experimenting with real sensors. Therefore, developing a method to simulate LiDAR sensors within NeRFs could significantly aid in selecting the correct sensors for specific applications. To understand how a LiDAR sensor can be simulated within a NeRF, it is essential to delve into the principles of distance measurement and the role of constants like the velocity of light. 

An important area for sensors is measuring, especially distance measuring. In order to facilitate measurement, it is necessary to employ a constant value that is universally applicable and accessible to all. One such fundamental physical constant is the velocity of light, which can be used to calculate the distance between two points in space by determining the time required for light to travel between them. This concept forms the basis of the measurement principle known as "time-of-flight" (ToF) #cite(<hahne_real-time_2012>, supplement: [p.~11]). One prominent example of a time-of-flight sensor is Light Detection and Ranging (LiDAR). They are different types of LiDAR. One type of LiDAR–which is the focus of this study—is the use of lasers that are directed at an object or scene and then measure time it takes for the reflected light to return to the receiver, allowing it to generate a 3D or 2D image with spatial and depth data @noauthor_sick-lidar_nodate. It is difficult to estimate which LiDAR could be correct for each given application, highlighting the need for effective simulation methods. By focusing on simulating LiDAR sensors within a NeRF, this thesis aims to provide a tool for assessing the suitability of various LiDAR technologies in a controlled virtual environment only with images from the scene.

This approach could offer significant advantages in terms of safety, cost, and efficiency when determining the most appropriate sensor for specific industrial applications. To effectively simulate LiDAR sensors in a virtual environment, advanced methods for 3D scene representation are required.

It is difficult to measure the real world. Cameras can help, but they only show 2D scenes. Using a LiDAR sensor to measure the real world creates a 3D point cloud what is more difficult to understand. One way to create a 3D scene from 2D images is with NeRF.

In 2020, #cite(<mildenhall_nerf_2020>, form: "author") introduced NeRF as an AI-based approach for generating 3D representations of scenes from 2D images. One of the key advantages by using NeRF is the low memory usage and the photo-realistic representation. With the use of pipelines, it is also easy to use NeRFs, even for untrained users @schleise_automated_2024. The central idea behind NeRF is to utilize a neural network, rather than techniques that use a grid or a list system to determinate each point and its corresponding values within a 3D scene.
#right_sign
Like other neural networks, NeRF suffers from the disadvantage of being a "black box" @la_rocca_opening_2022. This makes it difficult or even impossible to fully understand the computations taking place, as the neural network provides limited interpretability. As a result, it is challenging to determine positions, distances, or lengths within the scene. Unlike methods, which typically store scene data in grids or lists, where each pixel within the scene is explicit know, NeRF only compute RGB and density values from given coordinates. Depending on the method which the neural network is trained, the distances and coordinates from the original scene are unknown.\
To illustrate this concept in a simplified manner, each scene in NeRF can be conceptualized as a "memory" of the neural network, which learns to generate RGB and density values from a multitude of perspectives and positions by minimizing the discrepancy between the original image and its internal representation.

#figure(image("images/related_work/lidar_vs_nerf.png"), caption: [Simplified illustration depicts the concept of LiDAR and the rationale behind this thesis. While a time-of-flight sensor allows light to pass through and measures the time it takes for light to travel to an obstacle and reflect back to a receiver (illustrated on the left), a NeRF does not actually cast rays within the scene. The challenge is to identify the sample point that is closest to the obstacle and compute the distance between the origin and the sample point, as the coordinates of the origin and the sample point are within the scene.])
The employment of multiple lasers facilitates the generation of a point cloud representation of the surrounding environment. These sensors are specifically designed for real-time applications. However, the utilization of a NeRF to synthesize data in real-time remains a challenge and is currently not possible. The computation of distance between two points within a NeRF scene an generating a point cloud necessitates the completion of several steps.
#v(2mm)
#strong[Determine the points:] This represents a significant challenge with regard to implementation. As mentioned, in NeRF are the coordinates and scale are implicit. This discrepancy in scale and coordinates between the original scene and the NeRF scene presents a challenge. Although this implementation assumed that the local origin in the global scene could be set manually with a visible object within the scene that is movable in runtime. The primary issue lies in determining the appropriate distance between this local origin and each potential obstacle that a ray might encounter. As in @volume_sampling described, the density value increases close to an obstacle. To address this issue, multiple methods are developed and tested to estimate the sample point closest to the obstacle by recognize the increasing density value.
#v(2mm)
#strong[Casting rays:] Once the origin within the scene has been established, rays must be cast in order to obtain the requisite distance measurement and point cloud reflection. The rays should be dependent on the local origin and not on the scene perspective, which are typically employed in NeRF for scene representation.
#v(2mm)
#strong[Distance measurement:] To measure the distance between two points, the Pythagorean theorem is applied. A key challenge, however, lies in the fact that the scale within the scene does not correspond accurately to that of the original scene, particularly when camera parameters are estimated rather than precisely known during training. Due this issue, a reference object within the scene can used for scale calibration.
#v(2mm)

This implementation can be employed for the purpose of measuring distances within a given scene, in a manner analogous to that of an artificial LiDAR. Additionally, it can be utilized for the precise measurement of distances between two points within the scene. Furthermore, the implementation allows for the use of a reference object within the scene, which is used to compute a scale factor that is then applied to the measured distances. This method also can used to mapping closed areas. \
#v(2mm)
In order to implement the method, a technique is employed which enables the detection of the density value at a given point within the scene, as well as the calculation of the distance between the origin and the points. NVIDIA Omniverse is utilized for the generation of images, the management of camera parameters and as a means of comparison with a simulated LiDAR. Meanwhile, Nerfstudio with Nerfacto is employed for the creation of the NeRF and the implementation of the proposed methods.
#pagebreak()
#pagebreak()

= Related Work <related-work>
To understand the implementation and the associated challenges, this chapter provides a deeper dive into the fundamental principles of Neural Radiance Fields (NeRF). Additionally, it highlights similar works that combine NeRF with LiDAR systems, offering a comprehensive context for the methods and contributions discussed later in this thesis

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
Using marker for pose estimation has the benefit of knowing the scale and the camera positions within the scene. For testing it, an dataset from Simon Bethke are used. He produce a tutorial (Title: Tutorial - Gaussian Splatting, Platform: YouTube, Channel: Simon Bethke) and uses AprilTag Scale Markers for pose estimation. On the other hand, it is a higher effort to use these markers within the scene. By using this markers, distance measuring is also possible without reference object.

#figure(
  image("images/introduction/pose_marker.png", width: 50%), caption: [A AprilTag Scale Marker within the garage scene from Simon Bethke.The distance in this scene are 0.0704m The correct size of this markers are 7x7cm.]
)

=== Neural Network Architecture <neural-network-architecture>
The MLP consists of nine fully connected layers: eight layers with 256 neurons each and one with 128 neurons. It uses the #emph[Rectified Linear Unit] (ReLU) as the activation function, a non-linear function that leaves positive inputs unchanged while setting negative inputs to zero.

After capturing multiple images of a scene from different perspectives and obtaining the extrinsic and intrinsic camera parameters, the neural network uses the pixels from each image to cast rays into the scene. These rays are divided into samples (points) at specific intervals. Each point has its own coordinates in the scene and a direction derived from the ray.

While the densities $sigma$ are determined by the exact position in space, the RGB values $c$ also requires the direction $theta , phi.alt$. This is important for multi-view representation because viewing the same position from different angles can reveal different content. This technique allows for the correct rendering of reflections and lighting effects depending on the perspective. To achieve this, the first eight layers of the MLP process the 3D coordinates $X = x , y , z$, outputting a density value and a feature vector. This feature vector is then combined with the viewing direction as polar coordinates $theta , phi.alt$ (the pixel perspective), and passed into the final layer, which returns normalized RGB color values @mildenhall_nerf_2020. Formally, this function can be represented as: \
\
$ X , theta , phi.alt arrow.r R G B , sigma $ \
This function is important for solving a challenge of this thesis. To estimate the point on an object, a method is needed to calculate the correct density when the point is as close as possible to the obstacle.
#figure(image("images/related_work/nerf_rays.png", width: 60%), caption: [Illustration of rays in NeRF. Casting a ray for each pixel from an image, which includes the 5D input direction $theta, phi$ and sampling points as a coordinate $x, y, z$. The direction and the coordinates are the input for the neural network, which outputs the density $sigma$ and the RGB value.])<fig:nerf_rays>

=== Volume Sampling in Neural Radiance Fields <volume_sampling>
A critical component of NeRFs is the sampling strategy employed to evaluate the radiance field along camera rays. This section discusses the stratified sampling approach used in NeRFs, explaining how it selects sample points along rays, computes transmittance to detect potential object intersections, retrieves RGB and density values, and ultimately renders the scene from a chosen perspective. This section is important for the understanding how NeRF renders the scene and set the sampling points along a ray. Which is needed to approximate a point closest to an obstacle. \
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

The neural network approximates the spatial distribution of objects and surfaces in a scene and can then deduce which rays are reflected from which points and the resulting distances. Unlike other simulations, this approach is not based on explicit physical models, but on a data-driven reconstruction that adapts directly to the observed reality. This reduces the need to fully model the complex physical and optical interactions in the real world.

== Black Box <black-box>
As NeRF is a feed-forward artificial neural network (ANN), it suffers from the disadvantage of being a black box. Interpreting how the model (a trained ANN that is used to approximate a function of interest) uses the input variables to predict the target is complicated by the complexity of ANNs @la_rocca_opening_2022. Even those who create them cannot understand how the variables are combined to produce predictions. Black box prediction models, even with a list of input variables, can be such complex functions of the variables that no human can understand how the variables work together to produce a final prediction @rudin_why_2019. To interpret NeRFs computation, which are only complex functions, as a coordinates, it is not possible to efficiently and quickly define the values while the ANN is computing. It limits the ability to understand how exactly the calculations are done and limits the understanding of the coordinates to the result. One challenge in this implementation is determining the position of a point on an obstacle, which is not feasible for exact values. To address this issue, it is essential to comprehend the underlying computation of NeRF densities. Given that NeRF is a black box, it is only possible to analyze the results, not the computation itself. This restricts the possibility to understand or manipulate for the approximation problem.

== Similar Work
This section will show other studies that are working on the challenge to synthesize LiDAR sensors within a NeRF scene, and what are the uniqueness of my work.

=== NeRF-LiDAR <nerf-lidar>
#figure(image("images/nerf_lidar.png", width: 70%), caption: [The test scene in NVIDIA Omniverse (A) and the resulting point cloud as plot (B) from this application (vertical FOV: 360°, angular resolution: 0.5; horizontal FOV: 100°, angular resolution: 1.8). NeRF-LiDAR from Zhan et. al. (C) and a comparison with a real LiDAR (D).])

#cite(<zhang_nerf-lidar_2024>, form: "prose") also generate LiDAR data by employing real LiDAR data to compute precise 3D geometry as a supervisory signal to facilitate the generation of more realistic point clouds and achieve good results. In contrast to the approach taken by Zhang et al., this thesis synthesizes LiDAR data without the use of additional input data for NeRF. The use of LiDAR data for training offers the advantage of improved results and a more accurate understanding of the scene under consideration. However, this approach does not fully capture the nuances of LiDAR-specific properties, such as ray drop, luminescence, or the permeability of the medium through which the light propagates. While Zhang's research offers valuable insights into specific scenes, my approach may not be as accurate but is more accessible and can be applied to any scene that a NeRF can generate.

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
This chapter explains Nerfstudio and NVIDIA Omniverse and why I used them for this thesis.

== Nerfstudio <nerfstudio>
Nerfstudio is a PyTorch framework. It contains plug-and-play components for implementing NeRFs. It uses tools for real-time visualization and introduces how to work with and implement NeRFs @tancik_nerfstudio_2023. Nerfstudio is used for using and editing Nerfacto and rendering the scene on the front-end. While Nerfstudio's common use is to implement new NeRFs, this work use a modified version of Nerfacto.

== NVIDIA Omniverse <nvidia-omniverse>
Since the computation from distance in a NeRF depends on the camera parameters, it is easier to use graphical software where the camera values and position can be set and extracted in a virtual environment. For NeRF, it is not necessary to use real images, but for the comparison with a real LiDAR, software is needed that can simulate a LiDAR sensor. One tool that embeds and LiDAR sensors, such as the Sick PicoScan 150, is NVIDIA Omniverse. \
NVIDIA Omniverse is powered by NVIDIA RTX and is a modular development platform of APIs and micro services for creating 3D applications, services and to break down data silos, link teams in real time, and produce physically realistic world-scale simulations with tools like Isaac Sim @nvidia_nvidia_nodate.

#figure(image("images/implementation/omniverse.png", width: 80%), caption: [#text(fill: blue, [Activity diagram how distance measuring and point cloud creating are execute in this implementation.])])

#pagebreak()

= Implementation <implementation>
The goal of this project is to define an origin within a NeRF scene representation and utilize it for distance measurement and point cloud generation within the scene. To achieve this, the implementation involves several steps, which are detailed in the following chapters.

#figure(image("images/implementation/first_shadow.png"), caption: [Activity diagram how distance measuring and point cloud creating are execute in this implementation.])<act_dig1>

The principal process is illustrated in @act_dig1. Once the model has been trained, the Nerfstudio 3D visualization (Viser) with LiDAR environment can be initiated with the CLI command #strong[#emph[ns-view-lidar]] and standard Nerfstudio flags. This action opens the frontend, which allows the user to add and position a frustum within the scene which represent the local origin. Upon clicking the button in the frontend, a Nerfstudio camera object is generated, and a resolution is established in the backend. The specified densities are calculated and then transmitted to the frontend.

Used hardware:

#table(
  columns: 2,
  stroke: none,
  [GPU:], [NVIDIA GeForce RTX 4060 Ti (16GB VRAM)],
  [CPU:], [Intel i5 13500 with 14 cores, 20 threads and 4.8 GHz],
  [RAM:], [128GB],
  [OS:], [Windows 11]
)

To set up the installation, an Anaconda environment is created with the necessary dependencies. These include Python version 3.8 or higher, Visual Studio 2022 for the C++ Build Tools, and CUDA (Compute Unified Device Architecture) from NVIDIA, which enables developers to write programs that execute threads in parallel on the GPU. Additionally, the tiny-cuda-nn library is used for training and inference of small neural networks on NVIDIA GPUs using CUDA. The torchvision library, part of the PyTorch framework specifically designed for computer vision tasks, is also included. It offers a collection of pre-trained models, data processing functions, dataset classes, and utility functions to simplify working with images.

== Scene Capturing
For the scene capturing, multiple images from the scene are required. Every important part of the scene should be captured multiple times from different angles, positions and heights. The use of a large depth of field (small aperture) is recommended to ensure that all parts of the image are in focus. Motion blur and depth of field can lead to errors in the reconstruction and should be avoided. Consistent lighting conditions in all shots are important, as changing lighting can confuse the model and lead to inconsistencies.

== Parameter creating
As mentioned in @positioning, intrinsic and extrinsic camera parameters are crucial for this implementation. Colmap is included in Nerfstudio, which then creates the necessary transforms.json file with the parameter for the training by using the CLI command #emph[ns-process]. However, in my tests, the results from Colmap within Nerfstudio were worse than when using Colmap as a standalone program outside of Nerfstudio. In this case, Nerfstudio can also create the transforms.json file using the #emph[ns-process] command but with additional tags and the path to the Colmap sparse data. It is recommended to obtain the camera poses directly without pose estimation. In this scenario, a transforms.json file must be created manually for the training.  

Excerpt from the transforms.json file:

 #figure(
  align(left,
    text(size: 10pt,
    pseudocode-list(booktabs: true)[
      + {
        + "w": image_width,
        + "h": image_height,
        + "fl_x": focal_length_x,
        + "fl_y": focal_length_y,
        + "cy": principal_point_x,
        + "cy": principal_point_y,
        + "k1": radial_distortion_coefficient_1,
        + "k2": radial_distortion_coefficient_2,
        + "p1": tangential_distortion_coefficient_1,
        + "p2": tangential_distortion_coefficient_2,
        + "camera_model": e.g OPENCV or PINEHOLE
        + "frames": [
          + {
            + "file_path": "path/to/image_1",
            + "transform_matrix": [
              + [r11, r12, r13, px],
              + [r21, r22, r13, py],
              + [r11, r12, r13, pz],
              + [0, 0, 0, 1],
          + },{
            + file_path: "path/to/image_2",
            + ...
        + }
  ],), ),
  caption: [The transforms.json used from Nerfstudio with intrinsic and extrinsic camera parameter. Nerfstudio allows only one camera for the training.]
)

== Training
This implementation focus on using Nerfacto with Nerfstudio. Nerfstudio has multiple types of Nerfacto variant with different memory usage which are needs different time to train the scene. The training can started with images and a transforms.json for camera parameter with the CLI command #emph[ns-train nerfacto-[TYPE] --data YOUR_DATA] in the anaconda shell. The training for the lowest Nerfacto for the scenes needs approximately 15 to 20 minutes depending how many images are used and with the used hardware described earlier. For the nerfacto-big, the second largest model, the time increased to approximately 4 hours to 5 hours. The largest model (nerfacto-huge) needs 7 hours to 9 hours to train the scene.

== Frontend Configuration
The goal is to set a frustum in the frontend which is used as a local origin for the ray casting within the scene. To shift an rotate the frustum, a frontend UI are implemented.

#figure(image("images/implementation/frontend.png"), caption: [Possible user interactions in the frontend (A). Current scale factor which represents how distances in the scene are scaled before or after calibration with a reference object in the scene (A1). Setting for point cloud within the scene to change the color, the size or the maximal distance how point clouds are represented (B). Positioning of the frustum for the local origin Point cloud (C). Resolution settings for individual and precise measurement (D). For synthetic Sick LiDAR sensors, the vertical and horizontal angular resolution can be set. So it does not depend on hardware characteristics for the ability to use a reference object within the scene to calibrate the NeRF distance (E). Precise measuring buttons to define two points within the scene and the calibration modal to calibrate the scale factor for distance measuring (F). Delete precise measuring points or set scale factor to 1 in editing (A2).]) <fig:frontend>

After positioning the frustum, the user may select either a pre configured Sick LiDAR sensor or an individual point cloud. Individual point clouds afford greater flexibility with regard to angular resolution and the quantity of rays. The frontend enables the measurement of two points in the scene, where the frustum is employed to generate the points and to calibrate the scale factor for distance measurement. This may be achieved by utilizing a reference object in the scene with a known size to calibrate the scale factor.
=== Viser
Viser is a three-dimensional web visualization library utilized within the Nerfstudio framework. The library is utilized for the rendering of both the NeRF scene and the graphical user interface (GUI). The Viser library was utilized to develop the frontend user interface (UI), the point clouds within the scene, and the frustum for the pose. However, Viser is constrained in its functionality, which presents challenges or even impossibilities when attempting to create user interfaces that adhere to current standards for user experience, personas, and user-centered design. While nerfstudio utilizes the right-handed frame convention (OpenGL), viser employs left-handed (OpenCV).


// The objective of this project is to establish an origin within a NeRF scene representation and utilize this origin for distance measurement and point cloud creation within the scene. To achieve this, the implementation requires a frontend for setting the origin and resolutions, visualizing the scene, and parsing the data within the scene representation. Nerfstudio is employed for this implementation (see @nerfstudio).


=== User Interactions
This section presents a comprehensive overview of the user's available interactions in the frontend.
For each method, the user is able to generate a point cloud with the uses method for distance measuring, plot the point cloud, which is recommended to illustrate only the point cloud without the scene and show all rays without the collision detection within the scene.

Due to the constraints of limited resources and render time, two distinct types of point clouds have been implemented. A non-clickable point cloud is provided for visual demonstration purposes only. In contrast, the clickable point cloud allows users to click on each point, and the distance from the origin to that point is displayed. In the non-clickable method, the point cloud itself is an object in the scene. In contrast, in the clickable point cloud, each point is a single object in the scene. The clickable point cloud is considerably more resource-intensive and time-consuming than the non-clickable method.

==== Synthesize Sick LiDAR
To illustrate synanthic LiDAR-Sensors, two LiDAR demonstrations are presented, utilizing the Sick LiDAR sensor (PicoScan 100 and MultiScan 100). While the angular resolution of this sensor is fixed, the NeRF implementation is more dynamic. The original resolution is displayed in the interface, but the user may select a differential option. This is particularly relevant for the MultiScan, which has a vertical angular resolution of 0.125 for 360°. The computation of these rays is a time-consuming process. The settings for each LiDAR demonstration are saved in a JSON file, and each demonstration is loaded dynamically in the front end. Additional demonstrations and capabilities can be incorporated in future implementations.

==== Individual Measuring
While the demonstration of LiDAR sensors has defined angular and resolution parameters, users may also employ the individual measurement. This method allows for the independent adjustment of vertical and horizontal resolution, as well as the number of rays. This functionality enables the comprehensive scanning of an entire scene from a single origin or measuring the distance from a single ray.

==== Precise Measuring
This implementation allows the user to make precise measurements between two points within the scene. To do this, the user can generate a clickable point cloud in any way. After clicking on the "Measure Point" button on the frontend (see in @fig:frontend (F)) and then on a clickable point on the point cloud, this coordinates are saved. By setting the second measure point, the distance between them two points are displayed within the scene. If the poses for the camera are estimated, the scale factor should be calibrated before distance measuring. Since a NeRF scene is a virtual scene, this functionality also works through obstacles.

==== Calibration
If the camera parameter are unknown, it is necessary to calibrate the scale factor within the scene. For this, the precise measurement includes a possibility for calibrate this scale factor. For the calibration, an object in the scene is needed where the size of this object are known. In best case, this object is predestined for measurement with colors, patters and forms which the ANN can recognize easily.

#figure(image("images/implementation/known_object.png", width: 200pt), caption: [Self-made reference object with a defined size and different patters for the ANN to recognize the object within the scene.])

To use this calibration, the user can set the two measuring points as described in precise measuring. Then click on "Open Calibrate Modal" (see @fig:frontend (F)), which opens a modal in which the distance from the reference object (the two points) in relation to the correct size of the object can be used to calibrate the scale factor. After this calibration, each measurement will be calculated with this scale factor. This can be easily undone.

// #figure(image("images/implementation/second.png", width: 300pt), caption: [Frontend implementation as activity diagram. After receiving the frustum data from the frontend, the user can choose between Sick LiDAR, individual measurement and calibration. This returns the LiDAR resolution and activates the process to obtain the densities in the backend (illustration by author).])

== Backend <backend>
This chapter explains the backend part of the implementation. The focus is on the components necessary for this implementation rather than providing a complete overview of the entire Nerfstudio backend. A custom CLI command, #emph[ns-view-lidar], is created for the viewer after training. This command generates files that are independent of the original Nerfacto backend build. These files are modified copies of the original files. The main file containing most of the code for this implementation is #emph[render_state_machine_lidar.py], which is a copy of #emph[render_state_machine.py].

After launching the viewer in the frontend using the #emph[ns-view-lidar] command within the Anaconda environment, a frustum with an individual color can be created in the frontend. This action also opens the UI, as shown in @fig:frontend. The code for this UI resides in the #emph[generate_lidar_gui] method in #emph[render_state_machine_lidar.py].

Once the frustum is positioned in the frontend and a button for creating a point cloud or measuring distances or point cloud creation is clicked, the main method of this application, #emph[generate_lidar], is invoked. Depending on the button clicked, this method generates a virtual camera object from the frustum’s origin within the scene. The camera object is created with the following parameters: camera_to_world, fx, fy, cx, cy, width, height, distortion, and camera type. This object is then used to calculate all densities along the rays and their corresponding locations, which are subsequently utilized for distance measurement, plotting, and point cloud creation.

Since Nerfstudio is created for real perspective views and this method use the perspective scene reconstruction method from Nerfstudio, a potential 360 degree view are not provided in Nerfstudio, which is necessary for a 360 degree LiDAR-Sensor. Therefore a additional camera type for Nerfstudio are implemented. This camera type are implemented in the Cameras class:

$ d_("ij") = - mat(delim: "(", cos(phi_j), sin(theta_i); sin(theta_i); cos(phi_j), sin(theta_i)) $


// - Kurze Beschreibung, wie das Backend verläuft. Nur, dass eine Kamera generiert wird. Diese dann durch verschieden Schritte das ANN anspricht und anschließend eine Liste mit Daten (Density, positions) zurück gibt.
// - Die Änderungen die ich hinzugefügt habe, z.B: das hinzufügen eines eigenen Kamertyps, da Nerfstudio auf echte Kameras ausgelegt ist, und nicht auf LiDAR mit 360 Grad.

== Scene recognition
NeRF as an artificial neural network learns from differences within images. These differences can be in terms of color, texture, shape, or lighting variations. When a scene contains homogeneous areas—regions with uniform color and texture—NeRF faces challenges in accurately reconstructing these areas spatially.\

In the Save-Robotic scenario from Sick, this issue was observed with the floor. Although NeRF was able to render the floor correctly from a perspective view, the spatial reconstruction was not precise enough for the intended distance measurements. This problem arises from two main factors:\

- #strong([Homogeneity:]) NeRF learns to model a scene by mapping spatial coordinates $x,y,z$ and viewing directions $θ,ϕ$ to color values $"RGB"$ and density $σ$. It does this by minimizing the loss function, which represents the difference between the rendered outputs and the ground-truth images during training. \

  In homogeneous areas like a uniformly colored floor, there is a lack of distinguishable features or variations that can help the network associate different spatial positions with specific outputs. This absence of variation means that while NeRF can reproduce the appearance of the floor from known viewpoints, it may not accurately capture the spatial depth or geometry in these regions. Consequently, certain spatial measurements may not achieve the desired level of accuracy.

- #strong([Reflection:]) Reflective surfaces introduce additional complexity because they cause view-dependent appearance changes. In the case of a homogeneous and reflective floor, reflections can cause the floor to appear differently from various perspectives. While NeRF is designed to model view-dependent effects, the reflections on a homogeneous floor do not provide consistent spatial cues that correlate with actual geometric differences.\

  This inconsistency can affect the network's ability to learn an accurate spatial representation of the floor. The reflections introduce variations that are not indicative of spatial changes, potentially leading to inaccuracies in the reconstructed spatial properties. 

#figure(image("images/implementation/floor/floor_illustration.png", width: 70%), caption: [Expected ground rendering (A). Every pixel in the rendered scene is perspective. The spatial view depends on the environment, which is also perspective. The ground is scattered in different distances from the viewpoint. This makes it difficult to measure similar distances to the original scene (B).]) <fig:floor> 

As mentioned is the perspective view as expected. Due the issue to recognize the correct location of the floor resp. the pixel of the floor, the distances from the pixels are not correct. As shown in @fig:floor the pixels are scattered. \

The following illustrations depict various floor plans. The leftmost column presents an image from the scene in which point clouds are generated from an origin using the distance measurement algorithm that I have used. It is evident that in the first two examples, the original floor level is not recognized, except for the shadow of the box. Points outside the shadows are located below the floor level, which is more clearly illustrated in the middle image, a plot created for better visualization. The varied colors of the points indicate different distances for each point. When the colors of points within the shadow are similar to the surrounding colors, the floor is detected. The last column displays a graph representing multiple tests conducted in each scene. For these tests, 160 samples with 50 rays at different positions were used, as illustrated in @fig:floor_ex below.

#figure(image("images/implementation/floor/graph_example.png", width: 80%), caption: [Left image: Illustration for the floor test shown in the graphs below. In every scene are 50 samples taken from different positions. Each sample has 160 rays from the same origin. Right image: A good result of the graphs. Due the same origin of each sample, the distance of center of each graph should be smaller than the edges (exact at 1 meter). The vertical axis represents the 160 rays and the horizontal axis represents the distance.]) <fig:floor_ex>

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
  caption: [Left: Point cloud of the NeRF scene with gray ground and reflections. The points are only on the cube and the ground. Middle: Point cloud plotted for better recognition. Right: Plot described below without recognizable pattern. To illustrate the problems of spatial recognition by ANN from NeRFs by using a homogenous floor and reflections.]
) <fig:floor1>
  
The first example use a homogeneous gray floor with reflections. As @fig:floor1 illustrated, the left image shows some artifacts at the bottom of the image. This is due to the reflections in the original scene. The point cloud on the cube and the shadow on the floor are well visible. There are more points outside the shadow that are hard to see. These points are below the ground level.

This is better visible on the middle image which is the plot from the same position. 

The graph on the right image is difficult to evaluate and there is no recognizable pattern. Most of the areas below the frustum have no values because the distance is limited to 50 meters for better representation. 

The recognition of the floor depends on the neural network and is not comprehensible because it is a black box. It is important to note that the point cloud from the left and the middle image are similar but not the same. Both point clouds has the same origin in the same scene but different outputs. By using a neural network, even the same position and scene leads to different outputs. They are similar but not identical. The results of the graph use the same scene but not the same positions as the first two images. Therefore, they should not be compared directly.

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
  caption: [Left: Point cloud of the NeRF scene with gray ground and without reflections. The points are only on the cube and the ground but they more recognizable points outside the shadows. Middle: Point cloud plotted for better recognition. Right: Illustrate the problems of spatial recognition by ANN from NeRFs by using a homogenous floor but a better result than with refections]
) <fig:floor2>

The second example, shown in @fig:floor2, also has a homogeneous gray floor but without reflections. There are no artifacts on the bottom in the left image as in the first scene with reflections. This also happens in other test scenes where the reflections are removed. As in the first example, most of the points are on the cube and the shadow. 

As the middle image shows, there are also a few points outside these areas. We can see a kind of gradient under the frustum, which has more points than the first scene. 

The graph also has some parts without values. The right side of the graph still seems chaotic, but less random than the first example with a constant average distance of the points. This is a better but still not a good result because the edges shout have higher distances. The high peak on the left of graph depends on the less focus due to fewer images from the training (which is better to see in the fourth scene). The images are focused on the cube, the shadow, and the ground around it. Some of the samples are out of focus.

#figure(
  align(
    grid(
      columns: 3,
      column-gutter: 3mm,
      align: bottom,
      image("images/implementation/floor/checker.png", width: width),
      image("images/implementation/floor/checker2.png", width: width),
      image("images/implementation/floor/checker_graph.png", width: graph_width),
     ),
  ),
  caption: [Left: Point cloud of the NeRF scene with checkerboard pattern and without reflections. The points a visible the whole floor and not only on the shadow or cube. Middle: Point cloud plotted for better recognition. Right: The graphs shows a better result than the two scenes before.]
) <fig:floor3>

The third example, as shown in @fig:floor3, use a checkerboard pattern with different colors. The most points are on the floor and not only on the shadow and the cube. The use of this pattern led to a significantly better result as the scenes before, even if the checkerboard pattern is repetitive. The graphs shows in the middle an approximation to an good and recognizable pattern in average but still to chaotic for good distance measuring.

#figure(
  align(
    grid(
      columns: 3,
      column-gutter: 3mm,
      align: bottom,
      image("images/implementation/floor/interference.png", width: width),
      image("images/implementation/floor/interference2.png", width: width),
      image("images/implementation/floor/interference_graph.png", width: graph_width),
     ),
  ),
  caption: [Left: Point cloud of the NeRF scene with checkerboard and interference pattern, which makes the color of the floor more unique. The visibility of the points are similar to the previous scene. Middle: Point cloud plotted for better recognition. Right: The graph shows a good result. The left side are not as good as the left due to the focuses mentioned before. Distance measuring are good on this scene.]
) <fig:floor4>

The last virtual example (@fig:floor4) is the same pattern from the test scene I used. This scene also uses a checkerboard pattern, but with some interferences in between. This makes most of the areas on the floor unique. 

The left and middle image are similar to the previous scene. On closer inspection, the previous example seems to be better because of fewer outlines. The graph, on the other hand, shows a much better result than the previous example. As mentioned, the left side of the graph has more peaks than the right side, which is more in the focus of the images from the training. This scene shows how important scene recognition is and that homogenous colors leads to worse result in which distance measuring or point cloud generating does not lead to the desired results.

#figure(
  align(
    grid(
      columns: 3,
      column-gutter: 3mm,
      align: bottom,
     image("images/implementation/floor/center3.png", width: width),
     image("images/implementation/floor/center.png", width: width),
      image("images/implementation/floor/center2.png", width: width),
     ),
  ),
  caption: [The left image has four areas: Bottom right: The scene with grey floors and reflections from @fig:floor1. Top right: The scene with grey floors and without reflections from @fig:floor2. Top left: The scene with the checkmate pattern from @fig:floor3. Bottom left: The scene with the checkmate pattern and with interference from @fig:floor4. The middle images shows the plot from the left image for better illustration. The right image is also a plot of the same scene but from the side view. The Figure shows, that the images without checkmate pattern similar to each other but different to the images with checkmate pattern, which are also similar in a direct comparison.]
) <fig:floor5>

A direct comparison from each floor and from different angles. The difference between the gray pattern with and without reflection and the checkerboard with and without interference is not visible in these images.

#figure(
   grid(
    columns: 3,
    column-gutter: 3mm,
    align: center,
    image("images/implementation/floor/red.png", width:  150pt),
    image("images/implementation/floor/red2.png", width:  150pt),
  ), caption: [Real scene with red carpet. Left image: Reference object on a red carpet. Even with pattern on the carpet, the point cloud is only on the cube and the shadow. Left image: Plot for better illustration.]
) <fig:floor6>


@fig:floor6 shows a 3D scene representation from a real scene with my self-made reference object on a red carpet. Even with different pattern in the carpet, only this parts on the floor are recognized which are affected from a shadow.

#figure(
   image("images/implementation/floor/garage.png", width: 300pt),
    caption: [Real scene from the garage scene. Left image: It is to seen, that the point cloud only appears on this part who has different in the colors an pattern. Left image: Plot for better illustration.]
  ) <fig:floor7>

The point in the cloud point in @fig:floor7 are only on this position where the ground in the garage scene has different colors and pattern. This illustrates, that even NeRF scenes from grey industrial pattern with signs of use has problems to recognize the floor.

#pagebreak()
= Density Distance Algorithm <density-distance-algorithm>
A LiDAR sensor emits lasers in one direction and calculates the time it takes for the light to travel to an obstacle, reflect and return to the sensor. Because the speed of light is constant, this measurement is very accurate.
A NeRF Scene does not know any obstacle or coordinates. After casting a ray, it estimates the density volume along a ray based on Beer-Lambert-Law, which provides a description of the effects arising resulting from the interaction between light and matter @mayerhofer_bouguerbeerlambert_2020: 

== Single ray <single-ray>
It is important to understand the functionality of a single ray. Each ray returns a list of density values and their corresponding coordinates. Each value is computed on the Beer-Lambert method described above. \
NeRF can estimate new perspectives by learning a continuous and differentiable function of the scene that allows views from arbitrary angles. It effectively interpolates between the known perspectives using the learned 3D properties of the object. \
Due to the fact that each ray has infinite points along the ray, a sampling rate is needed (see @volume_sampling). If the estimated density value along the ray does not increase, a sampling point is set on the ray after a certain interval. When the density value increases, indicating that the ray has hit an object, both the density and the sampling rate increase. After the ray passes through the object, the sampling interval becomes larger again, as there is less relevant information behind the obstacle. \
For testing, a test scene is created with NVIDIA Omniverse. This test scene is built with a $1³$ m pattern to easily estimate distances and a color pattern for the ANN to better recognize the scene. To analyze the behavior of a single ray and its density values, several single rays are simulated in this scene with an exact distance of 1 meter to a wall within the scene.

#figure(image("images/Density_Distance_Algorithm/dda1.png", width: 100%, ), caption: [Sampling points along a ray. Constant sampling rate before collision with the object (A). Higher sampling rate due to the wall (B) and lower sampling rate with increasing distance (C). For better illustration, the sample points are smaller in B and larger in C.])

=== Accuracy <accuracy>
Since each ray has an unlimited number of points, it is necessary to use a specific sampling rate as described earlier. On the other hand, it is impossible to set the sampling rate so that an exact distance can be measured if the distance is not known. In reality, a tolerance value are set.
Therefore, the target is to obtain this sample point which is at closest to an obstacle.

To test this, several single rays are cast at a distance of 1 meter to the obstacle and the density value closest to 1 meter are plotted. For 11 different locations with different colors where the ray hits the wall, 50 rays are cast. The total average distance of all 550 rays are 1.000014739 meter which is a total deviation from approximately 15μm. The average density value on this exact point are 86.447. An interesting part is the difference between the different densities from the locations:
<tab:my_table>

#[
  #show table.cell: it => {
    if it.x == 0 or it.y == 0 {
      if it.body != [Average Density:] {
        set text(white)
        strong(it)
      }
    } else if it.body == [] {
      // Replace empty cells with 'N/A'
      pad(..it.inset)[_N/A_]
    } else {
      it
    }
  }
  
  #align(
    figure(
      table(
        gutter: 0.2em,
        inset: (right: 1.5em),
        columns: 4,
        stroke: none,
        fill: (x, y) =>
          if x == 0 or y == 0 { gray },
        [#strong[Location]],
        [#strong[Average Distance in m]],
        [#strong[Average Density Value]], 
        [#strong[Average Deviation]], 
        [1], [0.999911142], [20.3412], [0.000088858],
        [2], [0.999950498], [58.9121], [0.000049502],// 0.000049502
        [3], [0.999995336], [59.6347], [#highlight(fill: rgb(122, 205, 255), "0.000004664")], // 0.000004664
        [4], [0.999994195], [#highlight(fill: rgb(255, 174, 122), "389.1807")], [0.000005805],// 0.000005805
        [5], [1.000088757], [194.9974],[ 0.000088757], // 0.000088757
        [6], [1.000124663], [#highlight(fill: rgb(122, 205, 255), "1.6392")], [0.000124663],// 0.000124663
        [7], [1.000016545], [5.7899], [0.000016545],// 0.000016545
        [8], [1.00001], [11.2547], [0.00001],// 0.00001
        [9], [1.000156424], [102.6740], [#highlight(fill: rgb(255, 174, 122), "0.000156424")], // 0.000156424
        [10], [0.999937118], [96.3068], [0.000062882],// 0.000062882
        [11], [1.000124663], [1.63928], [0.000124663],// 0.000124663
        ),
    )) <tab:my_table> 
]


#let width = 55pt
#let height = 50pt

#align(center,
    grid(
    align: center,
    columns: 6,     // 2 means 2 auto-sized columns
    rows: 2,
    gutter: 2mm,    // space between columns
    image("images/Density_Distance_Algorithm/single_ray_table/1.png", width: width, height: height),
    image("images/Density_Distance_Algorithm/single_ray_table/2.png", width: width, height: height),
    image("images/Density_Distance_Algorithm/single_ray_table/3.png", width: width, height: height),
    image("images/Density_Distance_Algorithm/single_ray_table/4.png", width: width, height: height),
    image("images/Density_Distance_Algorithm/single_ray_table/5.png", width: width, height: height),
    image("images/Density_Distance_Algorithm/single_ray_table/6.png", width: width, height: height),
    [Location 1], [Location 2], [Location 3],[Location 4],[Location 5],[Location 6],
  )
)


#align(center,
    grid(
    align: center,
    columns: 5,     // 2 means 2 auto-sized columns
    rows: 2,
    gutter: 2mm,    // space between columns
    image("images/Density_Distance_Algorithm/single_ray_table/7.png", width: width, height: height),
    image("images/Density_Distance_Algorithm/single_ray_table/8.png", width: width, height: height),
    image("images/Density_Distance_Algorithm/single_ray_table/9.png", width: width, height: height),
    image("images/Density_Distance_Algorithm/single_ray_table/10.png", width: width, height: height),
    image("images/Density_Distance_Algorithm/single_ray_table/11.png", width: width, height: height),
    [Location 7],[Location 8],[Location 9], [Location 10], [Location 11]
  )
)

It should be mentioned that the color is not directly responsible for the density value but only indirectly. How good the scene, the objects, the spatiality can be recognized for the training, depends among other things on the color as shown in previous sections. Therefore, different colors and pattern leads to different results and density values. It also shows, that the density value is can not choose as an indicator to estimate the closest point at an obstacle what the illustrations below also shows.

=== Closest Sample Point

The following illustration different single rays. As in the previous tests, the different from the rays to the virtual wall are 1 meter. The plots shows a section of the whole ray with a red point. This point are the closest sample of the ray to 1 meter. Therefore, this point which would be the best result.

#figure(
  image("images/Density_Distance_Algorithm/closest_density_point/1_1.png"),
  caption: [Ray 1 overview. The red point left shows the sample closest to one meter and a high peak because of the obstacle detection from NeRF.]
) <fig:csp11>

#figure(
  image("images/Density_Distance_Algorithm/closest_density_point/2_1.png"),
  caption: [Ray 1 closer view. Detailed view of the same ray as in @fig:csp11. The illustration shows, that the high peak is more than 1 meter from the search density point away.]
)  <fig:csp21>
#figure(
  image("images/Density_Distance_Algorithm/closest_density_point/3_1.png"),
  caption: [Ray 1 more closer view. Density increase before the search sample point.]
)  <fig:csp31>

As @fig:csp11, @fig:csp21 and @fig:csp31 shows, it is difficulty to estimate the search sample point along the ray which are as closest as possible to an obstacle. The increase of the density, when an object is detected is not at the search sample point. Also shows this example, that the density can increase even before this point.

#figure(
  image("images/Density_Distance_Algorithm/closest_density_point/examples.png"),
  caption: [More illustrations for the sample point closest to 1 meter. Vertical axis are the density, horizonal axis are distance in meter. The density peak is more away from the search point. Smaller density increase before the search sample (A). Increasing and decreasing from the density (B). Good example for the density increase (C). It is important to mention, that the graphs are not normalized. The left image are always the whole vertical axis while the right image is only a detailed view of the closest point. Therefore, the hight of the graph should non compared to each other.]
)

Similar results as in the examples before in all 511 plots. This shows the arbitrariness from the density increase for a recognized obstacle and illustrates the difficulty to obtain the correct sample point along the ray closest to an obstacle. Each ray has his own properties.

== Method
For distance measuring it is necessary to obtain the sample point closest to the obstacle. As the previous section shows, the property of each ray are individual, which makes it difficult to use a universal method to obtain this sample point.
The focus for this method lies on the density value which are increased when the virtual ray hits an object within the scene. Multiple methods are tested.

=== Not used methods <methods>
For future researches, this section shows a short representation of this methods which not used in this implementation.

#strong[Threshold Method:] If the density value from a sample along the ray exceeds an given threshold value, these sample are used for distance calculation.

#strong[Different Method:] This method is similar to the first one, but the threshold should not be set by the user. While the first method sets an absolute threshold that is independent of each ray, the focus on increasing each density makes each ray independent of other rays.

#strong[Z-Scored Method:] In this method, the z-score is used to standardize the density values along each ray and identify points where the density deviates significantly from the mean.

#strong[Standardized Method:] In this method, the volume densities along each ray are standardize to account for global variations.

#strong[Accumulated Method:] In this method, the volume densities accumulate along each ray until the cumulative density reaches or exceeds a predefined threshold 

#strong[Maximum Density Change Method:] This method calculates the absolute density changes between successive sample points along each ray.

#strong[Average Method:] Compared to the previous method, where I identified the point along the ray with the maximum density change between consecutive samples, this method similarly calculates the absolute density differences between consecutive points along each ray.


// This section shows the methods for estimating the sample point closest to an obstacle. For this purpose, all rays thrown are analyzed and several methods are tested. The rays themselves are not modified.

// To simulate a NeRF scene, an origin in the scene is required, which can be added as a frustum within the scene. Depending on user input, this frustum simulates a ray within the scene and obtains the density values along a ray.

// The core idea is to estimate the density sample point $sigma$ at the position $x$ closest to the obstacle where the ray intersects, and then calculate the distance between the origin and this estimated point. Several methods are used and tested for this estimation.

// ==== Threshold method <simple-threshold-method>
// For each ray, I sample the volume densities $sigma(x)$ at discrete points along the ray path. By defining a threshold $sigma_(t h)$, which determines the density above which a relevant interaction is assumed, the algorithm searches for the first position $x_(t h)$ where $sigma(x_(t j)) gt.eq sigma_(t h)$. This corresponding position $x_(t h)$ is then used to determine the distance to the object. \

// While this method is commonly used for testing, it has two problems. First, the use of a threshold itself. Even though the use of a threshold is necessary in any method, it is a goal of this work that the user can use this implementation without defining a threshold because of the second problem: the definition of the threshold. As I mentioned before, the density value closest to the obstacle is arbitrary and different from ray to ray and scene to scene. Even a ray at the same position will give different values. While the test results are good with known distance, it requires the distance to calibrate $sigma_(t h)$, which makes it useless.

// ==== Difference method
// his method focuses on the increases in density along a ray. If $abs(Delta sigma) gt.eq Delta_(t h) $, where $Delta sigma = sigma_i minus sigma_(i-1)$ and $Delta_(t h)$ represents a threshold, then $sigma_i$ corresponding position $x_i$ is used for the distance calculation.

// This method is similar to the first one, but the threshold should not be set by the user. As demonstrated by @single-ray, the density value of the required density closest to the obstacle will vary from ray to ray, even if the origin and direction of the ray are the same. While the first method sets an absolute threshold that is independent of each ray, the focus on increasing each density makes each ray independent of other rays. Since the densities are too different, this method suffers from the same problem as the first method, but with better results and usable without threshold calibration.

// ==== Z-Score method
// In this method, I use the z-score to standardize the density values along each ray and identify points where the density deviates significantly from the mean. For each ray, I calculate the mean $mu$ and the standard deviation $s$ of the density values ${sigma_i}$. The z-score for each density value is calculated as:
// $z_i = (sigma_i - mu) / s$

// ==== Standardized method
// In this method, I standardize the volume densities along each ray to account for global variations. First, I compute the global mean $μ$ and the global standard deviation $s$ of all volume densities across all rays. For each ray, I standardize the density values $σ_i$ along the ray using $accent(σ, ~)_i=(σ_i -μ)/s$.

// Then I defined a threshold $Delta_(t h)$ that determines when a density is considered significant. Search the standardized density values along the ray and identify the first point $x_(i∗)$ at which $accent(σ, ~)_(i∗) gt.eq Δ_(t h)$ holds. This position is used to calculate the distance to the object.

// ==== Accumulate method
// In this method, the volume densities $σ_i$ accumulate along each ray until the cumulative density reaches or exceeds a predefined threshold $Sigma_(t h)$. For each sample point $x_i$ along the ray, the cumulative density $S_i =Sigma_(k=1)^i σ_k$ is calculated. The first point $x_(i∗)$ where $S_i gt.eq Sigma_(t h)$ is found. This position is used to calculate the distance to the object.

// === Maximum density change
// This method calculates the absolute density changes between successive sample points along each ray. Let $σ_i$ be the volume density at point $x_i$. The density change between points $x_(i-1)$ and $x_i$ is calculated as $Delta_(sigma_i) = abs(sigma_i -sigma_i -1)$. The index $i^∗$ at which the density change $Delta_(sigma_i)$ is maximized is determined: $i^∗ = arg max Delta_(σ_i)$. The corresponding position $x_(i^∗)$ is assumed to be the position where the ray hits the object, and the distance is calculated.

// ==== Average density change
// Compared to the previous method, where I identified the point along the ray with the maximum density change between consecutive samples, this method similarly calculates the absolute density differences between consecutive points along each ray. However, instead of relying solely on the maximum density change, I calculate the average density difference $accent(Delta_sigma, -)$ over the entire ray. I then define a dynamic threshold $Delta_(t h) = k * accent(Delta_sigma, -)$, where $k$ is a scaling factor.

=== Used method
Once the frustum has been positioned with respect to the origin and direction, the vertical and horizontal angular resolution has been obtained, and the button has been pressed to generate the point cloud within the scene, a Nerfstudio camera object will be created. Furthermore, all densities and locations along each ray will be saved in a list.
It should be noted that transmittance along a ray is defined in Nerfstudio as:
$ T(t) = exp(- integral_(t_n)^t sigma (r (s)) d s) $ Where $sigma (r (s))$ is the volume density along the ray depending on the position $r (s)$ on ray $r (t) = o + t d$.
Since the ray is scanned at discrete points, the integral can be approximated by a sum over the segments:\
#align(center,$T_n = product^n_(i=1) exp(-sigma_i ⋅ Delta_(s_i))$) where:\ 
- $sigma_i$ is the density on point $i$ \
- $Delta_(s_i)$ is the distance between point $i-1$ and $i$ \
- $T_o = 1$ the initial transmittance \

A collision is detected when transmittance $T_n$ falls below the threshold: $T_n gt.small T_(t h)$, where $T_(t h) = 10^(-200)$. A low threshold ensures that even the smallest density changes and faintly visible details are detected.

The integration as pseudocode:
#figure(
  kind: "algorithm",
  supplement: [Algorithm],
pseudocode-list(booktabs: true)[
  + *Function* find_collision_with_transmittance(ray_locations, ray_densities, threshold):
    + Set transmittance to 1.0  #text(fill: rgb("#555"), [/\/ Initial transmittance value ])
    + origin = frustum_origin

    + *For* each location, density in ray_locations[1:], ray_densities[1:]:
      + Set and calculate distance between origin and current location
      + Set and calculate delta_transmittance as exp(-density \* distance)
      + update transmittance by multiplying with delta_transmittance
      + *If* transmittance < threshold:
        + Return: distance, location, density #text(fill: rgb("#555"), [/\/ Collision found])
      + *End If*
    + *End For*
    + Return: No collision found  #text(fill: rgb("#555"), [/\/ If no collision is detected])
  + *End*
  ], caption: [The function simulates the passage of a ray through a medium that has different densities at different points. It calculates the transmittance (permeability) along the ray and determines whether the cumulative transmittance falls below a certain threshold value. If this is the case, it is assumed that a "collision" has taken place (i.e. the ray has been significantly attenuated).]
)
#pagebreak()

= Results

#figure(
  grid(
    gutter: 2pt,
    image("images/results/result1.png"), 
    image("images/results/result2.png"), 
    image("images/results/result3.png", width: 70%), 
  ),
  caption: [Point cloud generating, plot and distance measuring within different NeRF scenes. Pose estimation with marker (A), pose estimation with colmap (B) and knowing camera position with a virtual environment (C)]
)
In this chapter I will present some test with a test scene and a real scene with a refence object. It was not possible to test it with a real LiDAR.
== Test Scene
#figure(
  image("images/results/test_scene.png"), caption: [Images from the the test scene. Left image: Camera pose for the training of this test scene. Middle: The test scene as NeRF in Nerfstudio. Right: Point cloud within the test scene. The floor was good recognized by the ANN.]
)

To gain a more comprehensive understanding of this implementation, a trial was conducted in a test environment. The test scene was trained with known extrinsic and intrinsic parameters and with interference patterns for enhanced recognition due to the training. Given the known position, a distance of one meter in the original NVIDIA Omniverse scene is equivalent to one meter in the test scene. This facilitates the measurement of distance and composition with the original scene.

== Test
In order to test the hypothesis, 2560 rays were cast onto different surfaces at varying distances. At each position, 10 rays were cast, and the distance was increased by 0.5 meters up to a maximum of 4.5 meters, which represents the border of the scene. Thereafter, the position was changed to another surface.

#align(
  center,
  table(
      columns: 2,
      stroke: none,
      align: start,
      [Average Deviation:], [0.0001116 m],
      [Average Density:], [233.069]
  )
)

#[
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
      gutter: 0.2em,
        inset: (right: 1.5em),
        stroke: none,
        fill: (x, y) =>
          if x == 0 or y == 0 { gray },
      columns: 3,
      align: center,
      
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
]

The table illustrates the small deviations observed in the test, which evaluates the accuracy of distance measurements within the scene. The following graphs depict the average deviation of all samples in this test. The vertical axis represents the deviation of each ray from the expected distance, independent of the actual distance to the wall. The horizontal axis corresponds to the individual rays. In an ideal scenario, the graph would be a horizontal line at zero, analogous to the mathematical function $f(x)=0$, representing no deviation.

#figure(
  image("images/results/average_re.png", ), caption: [Average distance from all 2560 samples. The horizontal axis represents the samples, while the vertical axis indicates the distance. Each point represents a distance measurement, as explained below.]
) <fig:result_all>

#figure(
   grid(
    columns: 2,
    row-gutter: 2mm,
    column-gutter: 2mm,
    align: bottom,
    image("images/results/1.png"), image("images/results/1.5.png"),
    image("images/results/2.png"), image("images/results/2.5.png"),
    image("images/results/3.png"), image("images/results/3.5.png"),
    image("images/results/4.png"), image("images/results/4.5.png"),
  ), caption: [This Figure shows eight graphs for the test described below. Each graph has different distance to the wall. Except some high peaks (in 1.5m and 4.5m) the graphs are similar to each other, if we assume, that every ray cast leads to a different result. It also shows a good average overall and confirms, that a higher distance to an object has no different values]
) <fig:result>

While @fig:result_all shows all samples from the test described below, @fig:result illustrates the test from each distance to the wall. Except some outlines, the results are in average good and are independent from the distance to the obstacle. This test confirms that at least in the test scene distance measuring and point cloud generation similar to an simulated LiDAR are possible and in some cases computes better values in distance measuring than real LiDAR sensors.
#pagebreak()

= Discussion <discussion>
The primary question of this thesis is whether it is possible to synthesize LiDAR sensors within a Neural Radiance Field (NeRF) to determine the most suitable type of LiDAR for specific applications in a potential environment. Due to the complexity of the real world, accurately simulating a real LiDAR sensor is challenging. Similarly, understanding spatial relationships in 2D images for 3D scenes remains a significant hurdle. This implementation can generate point clouds and measure distances within a NeRF scene, but it is subject to certain restrictions and limitations.

As demonstrated, the complexity begins with creating the scene itself. The coordinates and scales within a NeRF are not equivalent to those in the original scene when pose estimation is employed. Even when a reference object is used, as implemented here, high-precision measurements are limited if intrinsic and extrinsic camera parameters are unavailable or cannot be computed. These restrictions depend on the specific use case and the quality of the NeRF model.

The second issue lies in scene recognition. While perspective views generally yield good results for NeRF scene generation, the lack of spatial knowledge remains a critical challenge. Homogeneous colors present an obvious issue; however, even in scenes with good differentiation, spatial inconsistencies can arise, leading to poor results.

By using or calculating camera parameters, distance measurements can be precise if the NeRF is focused on the area of interest, the scene is well-recognized, and the quality of the NeRF is high. This applies both to distance measurements and to point cloud generation within the scene.

LiDAR sensors themselves are not perfect. Distance measurements from LiDAR sensors can have deviations from real distances. Up to 20 cm at high angles of incidence @laconte_lidar_2019. Poorly reflecting or highly absorbing materials can affect the backscatter of the ray, resulting in inaccurate measurements. One study shows that the range of LiDAR sensors can be reduced by up to 33% under these conditions @loetscher_assessing_2023. Other influencing factors are complex terrain, measurement height, surface roughness and forest, atmospheric stability and half cone opening angle @klaas-witt_five_2022. These physical properties also a negative factor when trying to simulate a real LiDAR sensor. LiDAR sensors has many influences on the result, as is showable in this work. Distance measurements are only taken when an emitted LiDAR ray reflects off an object and returns to the sensor. Cases where ray do not return are called ray drop @mcdermott_probabilistic_2024. This ray drop needs physical correct calculation. As mentioned in @lidar-simulation, physical computation of LiDAR sensors are difficulty due the complexity of the real word. Factors such as scene brightness also play an important role.

Through the use of a NeRF with an ANN, some of these LiDAR-specific issues are mitigated in this implementation. However, for high-precision measurements and complex or large scenes, this implementation cannot fully simulate the behavior of real rays. For applications in industrial settings, where the selection of a suitable LiDAR sensor is needed for relatively simple scenes, this implementation offers a safe, cost-effective, and practical alternative.

== Outlook

=== Psyhsical Propeties
As mentiont, LiDAR results depends on physical properties. While it es difficult to simulate such properties it would be interesting if such computation can done with one ore multiple neural networks implemented in NeRF.

=== Medium of Velocity of Light
The normal use case of an LiDAR-Sensor are in Air where the different between the medium through what the light has pass through are similar to vacuum. The benefit of this applications is, that real rays are not needed ans so on, it can also be used for other mediums, like water or glass.

=== Gaussian splat
After 

#pagebreak()

= Conclusion <conclusion>
#pagebreak()
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
