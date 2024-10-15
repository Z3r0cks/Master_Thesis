#import "@preview/lovelace:0.3.0": *

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


#show cite: it => {
  // Only color the number, not the brackets.
  show regex("[a-zA-Zö\d.,.-]"): set text(fill: blue)
  // or regex("[\p{L}\d+]+") when using the alpha-numerical style
  it
}

#show ref: it => {
  if it.element == none {
    // This is a citation, which is handled above.
    return it
  }

  // Only color the number, not the supplement.
  show regex("[a-zA-Z\d.,.-]"): set text(fill: blue)
  it
}

#show outline.entry.where(
  level: 1,
): it => {
  v(12pt, weak: true)
  strong(it)
}

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
#v(15pt)
#strong(text([Synthesized LiDAR sensor data from \ Neural Radiance Fields], size: 25pt)) \

#v(40pt)

#let title_size = 16pt
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
          text([submitted], size: title_size), text([00.00.2024], size: title_size),
))

#pagebreak()
 
= Abstract
#set heading(numbering: "1.")
#lorem(500)
#pagebreak()
#outline(depth: 4, indent: 2em)
#set page(numbering: "1")
#counter(page).update(1)

= Introduction <introduction>
The Latin word "sentire" which means "to feel" or "to sense," is the etymological root of the term "sensor." It provides an accurate and concise definition of the fundamental nature of sensors: They enable machines to perceive and interpret the world around them. By allowing humans, through machines, to observe and measure aspects of the environment beyond human capability, sensors foster the development of technologies and machines that enhance our understanding of the world, physics, human behaviour, and more. They are, therefore, indispensable for technological advancement. \

In order to facilitate measurement, it is necessary to employ a constant value that is universally applicable and accessible to all. This constant can be utilised as a universal language to describe the universe, provided that it is known. One such fundamental physical constant is the velocity of light, which can be employed to calculate the distance between two points in space by determining the time required for light to travel between them. As an sensor, this concept can be described as radar (radio detection and ranging) principle: Send a signal and measure the time it takes for the reflection to arrive. The measurement of the time required for a signal to traverse a given distance is referred to as \"time-of-flight\" (ToF) #cite(<hahne_real-time_2012>, supplement: [p.~11]). To be correct, the velocity of light depends on the medium though which it travels, but this is not necessary for this thesis because the difference between the speed of light in a vacuum and in air is small (0.00029%, @arribas_indirect_2020) but it is important in other mediums such as glass or water, which have a greater effect on the speed, and this could be interesting aspect to future prospects of this implementation. \
One prominent example of a time-of-flight sensor are Light Detection and Ranging (LiDAR) sensors. LiDAR is a technology that uses a laser to target an object and then measures the time it takes for the reflected light to return to the receiver. This means LiDAR sensors can generate a 3D or 2D image with spatial and depth data to detect, measure, localize, and track objects in real-time with 360° resolution. @noauthor_sick-lidar_nodate. They are different types of LiDAR sensor, such Frequency Modulated Continuous Wave LiDAR which are often used in cars or solid-state solid-state LiDAR which has no moving parts. \

In 2020, Mildenhall et al. introduced an AI-based method for generating a 3D scene representation. This method is known as #emph[Neural Radiance Fields] (NeRF). One of the key advantages by using NeRF is the low memory usage and the photo-realistic representation. With the use of pipelines, it is also easy to use NeRFs, even for untrained users @schleise_automated_2024. NeRF can be extended with other methods, such as object recognition through integrated additional neural networks or scene manipulation. The central idea behind NeRF is to utilize a neural network, rather than techniques that use a grid or a list system to determinate each point and its corresponding values within a 3D scene. However one problems which NeRF also address are the ill-posed problem:

- #strong[Non-unique solution:] There is no clear way to reconstruct a 3D scene from limited 2D images.

- #strong[Sensivity:] Small changes in the input can have big changes in the output.

- #strong[Incompleteness of the data:] Not all information about the scene is known in the input data.

NeRF compensate this ill-posed problem by using a neural network which learns the context from the scene. \
\
Like other neural networks, NeRF suffers from the disadvantage of being a \"black box\" @la_rocca_opening_2022. This makes it difficult or even impossible to fully understand the computations taking place, as the neural network provides limited interpretability. As a result, it is challenging to determine positions, distances, or lengths within the scene. Unlike methods, which typically store scene data in grids or lists, where each pixel within the scene is explicit know, NeRF only compute rgb and density values from given coordinates without reference to the original scene. \


#figure(image("images/introduction/implicite.png", width: 100%), caption: [Comparison between implicit and explicit coordinates in NVIDIA Omniverse and a NeRF scene representation. The use of NVIDIA Omniverse to calculate the distance between two known points using the Pythagorean theorem (A). The coordinates as user input: $x'_a , y'_a , z'_a$ for the neural network $f_theta$ which interprets these coordinates by computing pixels in the space it learns from training (B). Illustration of the problem of getting a point closest to an obstacle in NeRF (C) (illustration by author).])<fig:impicit>


As illustrated in @fig:impicit it is challenging to obtain the requisite coordination to measure the distance between the two points. While the distance can be readily calculated in NVIDIA Omniverse due to the availability of known coordinates, the coordinates in a NeRF scene are dependent on the original images and their associated parameters. It is feasible to approximate the same coordinates as in NVIDIA Omniverse; however, due to the issue of infinite points in space, it is not possible to obtain the exact coordinates, which represents a significant challenge in the implementation of this approach with LiDAR sensors, which are highly precise. Nevertheless, even when it would be feasible to obtain the exact positions, the distance would not be accurate, given that the NeRF scene lacks any references regarding scale. This understanding also depends on the camera parameters. Another Problem is to obtain the closest point on an obstacle. A NeRF approximates the original scene by approximates the RGB and density values. These values are not exact representations of the scene. In the real world or in graphical software, I can simply use a pen or a click on an object because its coordinates are known and defined. However, in a NeRF, every pixel is only an approximation of an RGB value, influenced by the density value. This means I cannot, for example, simply take a pen and mark a specific point, because the exact position and properties of that point are not explicitly defined in the model. \
\
This thesis addresses the scientific questions of whether is it possible to synthesize a LiDAR sensor within a NeRF scene. It explores the advantages and disadvantages of this approach, the requirements and differences compared to the data from a simulated LiDAR sensor. \
The implementation can be employed for the purpose of measuring distances within a given scene, in a manner analogous to that of an artificial LiDAR. Additionally, it can be utilised for the precise measurement of distances between two points within the scene. Furthermore, the implementation allows for the use of a reference object within the scene, which is used to compute a scale factor that is then applied to the measured distances. \
In order to implement the aforementioned method, a technique is employed which enables the detection of the density value at a given point within the scene, as well as the calculation of the distance between the origin and the aforementioned point. NVIDIA Omniverse is utilised for the generation of images, the management of camera parameters and as a means of comparison with a simulated LiDAR. Meanwhile, Nerfstudio with Nerfacto is employed for the creation of the NeRF and the implementation of the proposed methods. \
\

= Related Work <related-work>
== LiDAR Simulation <lidar-simulation>
According to @zhang_nerf-lidar_2024, one common approach to LiDAR simulation involves creating a 3D virtual environment and rendering point clouds using physics-based simulations. This method allows for the generation of synthetic LiDAR data. However, Zhang et al. note that these virtual datasets often exhibit significant domain gaps when compared to real-world data, especially when used to train deep neural networks. The primary reason is that virtual environments cannot fully replicate the complexity and intricate details of the real world. Instead of using physical simulation of the real world, this work focus on understanding a neural network to compute the correct positions from densities in a 3D scene.

== Black Box <black-box>
As NeRF is a feed-forward artificial neural network (ANN), it suffers from the disadvantage of being a black box. Interpreting how the model (a trained ANN that is used to approximate a function of interest) uses the input variables to predict the target is complicated by the complexity of ANNs @la_rocca_opening_2022. Even those who create them cannot understand how the variables are combined to produce predictions. Black box prediction models, even with a list of input variables, can be such complex functions of the variables that no human can understand how the variables work together to produce a final prediction @rudin_why_2019. To interpret NeRFs computation, which are only complex functions, as a coordinates, it is not possible to efficiently and quickly define the values while the ANN is computing. It limits the ability to understand how exactly the calculations are done and limits the understanding of the coordinates to the result. One challenge in this implementation is determining the position of a point on an obstacle, which is not feasible for exact values. To address this issue, it is essential to comprehend the underlying computation of NeRF densities. Given that NeRF is a black box, it is only possible to analyze the results, not the computation itself.

== Neural Radiance Fields <neural-radiance-fields>
#cite(<mildenhall_nerf_2020>, form: "author") present a method for generating an AI-based 3D scene representation called Neural Radiance Fields (NeRF). The key idea behind NeRF is to synthesise novel views from limited input. The advantage of using a neural network is that it requires less memory than techniques that use lists and produces photo-realistic results. The main component is a neural network FΘ, implemented as a multilayer perceptron (MLP), which is trained using the positions and directions (perspectives of pixel in an images) from input images to approximate RGB values and compute the density σ at a given point in the scene. The density and RGB values are then used to generate a 3D scene through volumetric rendering.
#figure(image("images/nerf_image_mildenhall_1.png", width: 85%), caption: [Overview of the NeRF scene representation. Sampling the 5D input location and direction x, y, z, θ, ϕ (a) to feed the neural network (b). Then using volumetric rendering techniques to render the scene (c) and optimizing the scene with backpropagation since volumetric rendering
function is differentiable (based on Mildenhall et al. (2020), Figure 2) (illustration by author).])<fig:shift_image>

=== Positioning <positioning>
=== Camera Parameter <camera-parameter>
(Not sure if in need a camera parameter section.)

NeRF learns from images. To do this, most NeRF variants require the extrinsic and intrinsic camera parameters. These parameters are important for measuring distances in a NeRF scene because both the positions and the scale factor in NeRF depend on the image parameters. \
These parameters are the exact positions and orientation of each images used for the scene in form of a homogeneous 4x4 transform matrix: $ mat(
  delim: "[",
  r_11, r_12, r_13, p_x;r_21, r_22, r_23, p_y;r_31, r_32, r_33, p_z;0, 0, 0, 1,
) $

The image resolution, focal length, principle point, and distortion. Nerfstudio only allows one type of camera parameters. \
In an ideal case, as in @fig:pos, all cameras are evenly spaced around a center point and all camera positions are known. This center point thus acts as the center of gravity and the origin of the coordinate system of the NeRF scene. The NeRF architecture recognizes and learns from the camera positions and can derive the distances in the scene based on these positions. \

#figure(image("images/related_work/pos.png", width: 100%), caption: [Left-handed: Symmetric and known camera positions where the cendroid is in the centre of the scene. The
result would be the same coordinate system with the same distance in NeRF as in the original system. Right-handed: Only five of the eight cameras are used. The centre of gravity moves upwards. Note: The coordinates in the right image are from the original system. In NeRF the cendroid would be (0, 0, 0) and the image positions would also be different
(illustration by author).])<fig:pos>

For example, if the cameras are exactly one meter away from the center point, this distance is also correctly interpreted by the NeRF and used to reconstruct the geometry of the scene. In practice, however, perfect symmetry or known camera poses are rare because the original scene does not allow symmetric image creation or the technical availability is not given for image poses outside a virtual world. Without knowing the exact camera parameters, this is often determined using tools that use structure-for-motion methods such as colmap, which leads to errors in pose detection. Using exact coordinates is faster and more precise than using tools for pose estimation and a reference object within the scene for scale estimation. \
Training the model with known camera poses has the benefit that the model knows exact positions from the original scene and it reduces the possibility of making a calibration error with a reference object within the scene. Navigation and usability in the scene are also easier as the original coordinates can be used.

#figure(image("images/related_work/pose2.png", width: 100%), caption: [omparison of the two NeRF scene with same original scene and images but different parameter creation. While the origin in the scene with known parameters is the same as in the original scene (a), the scale as well as the position and rotation are different in the scene where the parameters are estimated with colmap (b). The distance without using a reference object for scale calibration (c) and with (d) for the NeRF scene with pose estimation. The distance without scale calibration in the scene where the parameters are known (e). The original distance is 1 meter. Note: The scale of the frustum in b is larger than in a, the frustum is not closer to the scene camera. (illustration by author).])<fig:pos3>

=== Neural Network Architecture <neural-network-architecture>
The MLP consists of nine fully connected layers: eight layers with 256 neurons each and one with 128 neurons. It uses the #emph[Rectified Linear Unit] (ReLU) as the activation function, a non-linear function that leaves positive inputs unchanged while setting negative inputs to zero.

After capturing multiple images of a scene from different perspectives and obtaining the extrinsic and intrinsic camera parameters (see #emph[]), the neural network uses the pixels from each image to cast rays into the scene. These rays are divided into samples (points) at specific intervals. Each point has its own coordinates in the scene and a direction derived from the ray (image direction).

While the densities ($sigma$) are determined by the exact position in space, the RGB values $c$ also requires the direction ($theta , phi.alt$). This is important for multi-view representation because viewing the same position from different angles can reveal different content. This technique allows for the correct rendering of reflections and lighting effects depending on the perspective (In der Safe-Robotic Szene von Sick wurden die Scheiben als Objekt erkannt: testen ob es hiermit zu tun hat). To achieve this, the first eight layers of the MLP process the 3D coordinates $X = x , y , z$, outputting a density value and a feature vector. This feature vector is then combined with the viewing direction as polar coordinates $theta , phi.alt$ (the image’s perspective), and passed into the final layer, which returns normalized RGB color values @mildenhall_nerf_2020. Formally, this function can be represented as: \
$ X , theta , phi.alt arrow.r R G B , sigma $ \
This function is important for solving the first problem of this thesis. To estimate the point on an object, a method is needed to calculate the correct density when the point is as close as possible to the obstacle.

#figure(image("images/related_work/nerf_rays.png", width: 60%), caption: [Illustration of rays in NeRF. Casting a ray for each pixel from an image, which includes the 5D input direction (Φ, Θ) and sampling points as a coordinate (x, y, z). The direction and the coordinates are the input for the neural network, which outsputs the density (σ) and the RGB value (illustration by author).])<fig:nerf_rays>

=== Volume sampling <volume-sampling>
Too render a scene, a function in NeRF is implemented which use 5D as input. While learning, the ANN use a discrete set of sampling allow a continuous scene representation. Too estimate the volume density along a ray.

== Nerfacto <nerfacto>
After the publication of NeRF 2020, many other researches on this method have been published to improve or extend the original NeRF method. One of them is Nerfacto @tancik_nerfstudio_2023, which takes advantage of several NeRF variants. While vanilla NeRF works well when images observe the scene from a constant distance, NeRF drawings in less staged situations show notable artifacts. Due this problem Mip-NeRF @barron_mip-nerf_2021 addresses this by adopting concepts from mipmapping and enabling a scale-aware representation of the radiation field. By introducing integrated position coding, areas in space can be coded more efficiently, leading to better rendering results and more efficient processing. This helps with generating images from different distances which is more realistic for capturing images. To reduce the time for training, Nerfacto use the Insant-NGP’s method @muller_instant_2022. Instead of feeding the input coordinates into the network directly (as with NeRF), the coordinates are processed using multiresolutional hash coding. Since the hash coding already captures many details, the actual MLP can be much smaller than with NeRF. The MLP is mainly used to process the features provided by the hash coding and to calculate the final values (color and density). Summerized, Nerfacto is faster and more accurate than NeRF and as the integrated NeRF variant from Nerfstudio (see #emph[]), which is used for the implementation, it is supported and worked on.

== NeRF-LiDAR <nerf-lidar>
@zhang_nerf-lidar_2024 also generate LiDAR data by using real LiDAR data to compute accurate 3D geometry as supervision to learn more realistic point clouds and achieve good results. While Zhang et al. uses real data, this work synthesise LiDAR data only with a NeRF and without any additional input. Vergleich zwischen meinem und LiDAR-NeRF?!

#figure(image("images/nerf_lidar.png", width: 70%), caption: [Comparison of synthesized LiDAR data with real measurements (illustration by author). (A) Synthesized LiDAR in a
test environment; (B) LiDAR simulation with NVIDIA Omniverse; (C) NeRF LiDAR results; (D) Data from a real LiDAR sensor
according to Zhang et al. (2024). #text([Eigene Bilder mit besseren Beispielen tauschen, max distance in B entfernen)], fill: red))])

= Tools <tools>
== Nerfstudio <nerfstudio>
Nerfstudio is a PyTorch framework. It contains plug-and-play components for implementing NeRFs. It uses tools for real-time visualization and introduces how to work with and implement NeRFs @tancik_nerfstudio_2023. Nerfstudio is used for using and editing Nerfacto and rendering the scene on the front-end. While Nerfstudio’s common use is to implement new NeRFs, this work use a modified version of Nerfacto.

== NVIDIA Omniverse <nvidia-omniverse>
Since the computation from distance in a NeRF depends on the camera parameters, it is easier to use graphical software where the camera values and position can be set and extracted in a virtual environment. For NeRF, it is not necessary to use real images, but for the comparison with a real LiDAR, software is needed that can simulate a LiDAR sensor. One tool that embeds and certifies LiDAR sensors, such as the Sick PicoScan 150, is NVIDIA Omniverse. \
NVIDIA Omniverse is powered by NVIDIA RTX and is a modular development platform of APIs and microservices for creating 3D applications, services and to break down data silos, link teams in real time, and produce physically realistic world-scale simulations with tools like Isaac Sim @nvidia_nvidia_nodate. Nvidia Omniveres use Unversal Scene Description (USD). \
Large amounts of 3D data, are generated, stored, and transmitted by pipelines that can produce computer graphics for movies and video games. Modeling, shading, animation, lighting, effects, and rendering are just a few of the many collaborating applications in the pipeline. Each one has a unique scene description that is suited to its own workflows and needs. USD is the first software that deals with exchanging and supplementing any 3D scenes @pixar_openusd_nodate.

= Implementation <implementation>
#text([Schauen ob beim Einstellen eine Aktivierungsfunktion bessere Ergbenisse zu erwarten sind, da keine Negativen Werte für die Density zu erwarten sind bzw ob das Netz hier überhaupt darauf zugreift. oder es nur Part des Trainings darstellt (out\_activation\=None in nerfacto\_field.py:144)], fill: red). \
\
The target is to set an origin in a NeRF scene representation and use this origin for distance measurement in form of a synthetic LiDAR sensor. To accomplished this, the implementation needs a frontend where the origin and resolutions can set, the scene can be visualise and a backend for parsing the data within the scene representation. For the implementation Nerfstudio is used (see @nerfstudio).


#figure(image("images/implementation/first_shadow.png", width: 100%), caption: [Activity diagram for the implementation (illustration by author).])<act_dig1>

The main process is shown in @act_dig1. After training a model, the Nerfstudio 3D visualisation (viser) with LiDAR environment can be started with the CLI command #strong[#emph[ns-view-lidar]] and standard Nerfstudio flags, which opens the frontend where a frustum can be added and positioned within the scene. After clicking on a button in the frontend, a Nerfstudio camera object is created, and a resolution will be set in the backend, which computes the given densities and sends them to the frontend.

== Viser <viser>
Viser is a 3D web visualisation library employed within the Nerfstudio framework. The library is employed for the rendering of both the NeRF scene and the graphical user interface (GUI). This library was employed to develop the frontend user interface (UI), the point clouds within the scene, and the frustum for the pose. However, Viser is constrained in its functionality, which presents challenges or even impossibilities when attempting to create user interfaces that adhere to current standards for user experience, personas, and user-centered design. While Nerfstudio utilizes the OpenGL frame convention, Viser employs OpenCV.

== Frontend <frontend>
The frontend is the user interaction area where a frustum with individual color and resolution can be created and positioned to generate points clouds for distance measuring.
#figure(image("images/implementation/frontend.png", width: 70%), caption: [Possible user actions in the frontend. Current scale factor for distance which represents how distances in the scene are scaled before or after calibration with a reference object in the scene (a). Point cloud settings for individual color or size of each point cloud (b). For synthetic Sick LiDAR sensors, the vertical and horizontal angular resolution can be set. So it does not depend on hardware characteristics (c). Ability to use a reference object within the scene to calibrate the NeRF distance (d) (illustration by author).])


To get a point cloud within the scene or as a plot, an user has to positioned the frustum. After this, the user can choose from a preconfigured Sick LiDAR sensor or individual point cloud. Individual point clouds are flexible with the angular resolution and the quantity of rays. #link(<frontend>)[\[frontend\]] shows the possibility to measure two points in the scene where the frustum is used to generate the point\(s) and to calibrate the scale factor for distance measurement where is a reference object in the scene with an known size can used to calibrate the scale factor (see ).

#figure(image("images/implementation/second.png", width: 300pt), caption: [Frontend implementation as activity diagram. After receiving the frustum data from the frondend, the user can choose between Sick LiDAR, individual measurement and calibration. This returns the LiDAR resolution and activates the process to obtain the densities in the backend (illustration by author).])

== Backend <backend>
The main part of the backend is to parse the given data from the frontend and compute the density values which are then send back to the frontend. The data from the frontend are the exact position ($x , y , z$) from the frustum and its orientation ($theta , phi.alt$) in the scene. Due the complexity of Nerfstudio it is relevant to show the process to compute the densities more detailed. \
After clicking on a button to create a point cloud

= Density Distance Algorithm <density-distance-algorithm>
A LiDAR sensor emits lasers in one direction and calculates the time it takes for the light to travel to an obstacle, reflect and return to the sensor. Because the speed of light is constant, this measurement is very accurate. #text(fill: red, [Text über die Genautigkeit von einem LiDAR sensor]) \
A NeRF Scene does not know any obstacle or coordinates. After casting a ray, it estimates the density volume along a ray based on Beer-Lambert-Law, which provides a description of the effects arising resulting from the interaction between light and matter @mayerhofer_bouguerbeerlambert_2020: 
$ T lr((t)) = e x p lr((- integral_(t_n)^t sigma lr((r lr((s)))) d s)) $ Where $sigma lr((r lr((s))))$ is the volume density along the ray depending on the position $r lr((s))$ on ray $r lr((t)) = o + t d$. \"The Function $T lr((t))$ denotes the accumulated transmittance along the ray from $t n$ to $t$, i.e., the probability that the ray travels from $t n$ to $t$ without hitting any other particle." #cite(<mildenhall_nerf_2020>, supplement: [ p.~6 ])

== Single ray <single-ray>
It is important to understand the functionality of a single ray. Each ray returns a list of density values and their corresponding coordinates. Each value is computed based on the Beer-Lambert method described above. \
NeRF can estimate new perspectives by learning a continuous and differentiable function of the scene that allows views from arbitrary angles. It effectively interpolates between the known perspectives using the learned 3D properties of the object. \
Due to the fact that each ray has infinite points along the ray, the sampling rate depends on time and density value. If the estimated density value along the ray does not increase, a sampling point is set on the ray after a certain interval. When the density value increases, indicating that the ray has hit an object, both the density and the sampling rate increase. After the ray passes through the object, the sampling interval becomes larger again, as there is less relevant information behind the obstacle. \
For testing, a test scene is created with NVIDIA Omniverse. This test scene is built with a 1³ m pattern to easily estimate distances and a color pattern for the ANN to better recognize the scene. To analyze the behavior of a single ray and its density values, several single rays are simulated in this scene with an exact distance of 1m to a wall within the scene.

#figure(image("images/Density_Distance_Algorithm/dda1.png", width: 100%, ), caption: [Left-handed. The test scene. Right-handed: Sampling points along a ray. The sampling rate is higher then before or after the ray hits the wall (illustration by author) #text([besseres bild rechts], fill: red).])

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
        [Ray 1], [0.999911142], [20.3412],
        [Ray 2], [0.999950498], [58.9121],
        [Ray 3], [0.999995336], [59.6347],
        [Ray 4], [0.999994195], [#highlight(fill: rgb(122, 205, 255), "389.1807")],
        [Ray 5], [1.000088757], [194.9974],
        [Ray 6], [1.000124663], [#highlight(fill: rgb(255, 174, 122), "1.6392")],
        [Ray 7], [1.000016545], [5.7899],
        [Ray 8], [1.00001], [11.2547],
        [Ray 9], [1.000156424], [102.6740],
        [Ray 10], [0.999937118], [96.3068],
        [Ray 11], [1.000124663], [1.63928],
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
it should be mentioned, that the color is not directly responsible for the density value. It is the accuracy itself, because the density value increases greatly on the positions where the ray is in front of, on, or inside an obstacle. That the object was sometimes recognized better and sometimes worse. In my test, samples with large density values are mostly closer to the object.

== Not used methods <methods>
This section shows the methods for estimating the sample point closest to an obstacle. For this purpose, all rays that are thrown are analyzed an multiple methods are tested. The rays itself are not changed.

To simulate a NeRF scene, an origin in the scene is required, which can be added as a frustum within the scene. Depending on the user input, this frustum simulates a ray within the scene and obtains the density values along a ray.

The core idea is to estimate the density sample point 𝜎 at position 𝑥 closest to the obstacle where the ray intersects, and then calculate the distance between the origin and this estimated point. For this estimation, various methods are used and tested.

=== Threshold method <simple-threshold-method>
For each ray, I we sample the volume densities $sigma(x)$ at discrete points along the ray path. By defining a threshold value $sigma_(t h)$, which determines the density above which a relevant interaction is assumed, the algorithm searches for the first position $x_(t h)$ where $sigma(x_(t j)) gt.eq sigma_(t h)$. This corresponding position $x_(t h)$ is then used to determine the distance to the object. \

While this method is commonly used for testing, it has two problems. First, the use of a threshold itself. Even if the use of a threshold in each method is necessary, it is a goal of this work that user can use this implementation without defining a threshold because of the second problem: definition of the threshold. As I mentioned, the density value closest to the obstacle is arbitrary and different from ray to ray and scene to scene. Even a ray at the same position results in different values. While the result with testing are good with known distance, it requires the distance to calibrate $sigma_(t h)$ which makes it pointless.

=== Difference method
This method is focused on the increases of the densities along a ray. When $abs(Delta sigma) gt.eq Delta_(t h) $, where $Delta sigma = sigma_i minus sigma_(i-1)$ and $Delta_(t h)$ represents a threshold, $sigma_i$ corresponding position $x_i$ is used for the distance computation.

This method is similar to the first method, but the threshold should not be set by the user. As demonstrated by @single-ray, the density value on the required density that is closest to the obstacle differs from ray to ray, even when the origin and direction of the ray are the same. While the first method set an absolute threshold value that is independent of each ray, the focus on increasing each density makes each ray independent of other rays. As the densities are too different, this method suffers with the same problem as the first method but with better results and useable without threshold calibrating.

=== Standardized method
In this method, I utilize the z-score to standardize the density values along each ray and identify points where the density significantly deviates from the mean. For each ray, I calculate the mean $mu$ and standard deviation $s$ of the density values ${sigma_i}$. The z-score for each density value is computed as:

#align(center, $z_i = (sigma_i - mu) / s$)
=== Z-Score method
In this method I standardize the volume densities along each ray to account for global variations. First, I calculate the global mean $μ$ and the global standard deviation $s$ of all volume densities across all rays. For each ray, I standardize the density values $σ_i$ along the ray using $accent(σ, ~)_i=(σ_i -μ)/s$.

Then I defined a threshold value $Delta_(t h)$ , which determines when a density is considered significant. The search of the standardized density values along the ray and identify the first point $x_(i∗)$ , at which $accent(σ, ~)_(i∗) gt.eq Δ_(t h)$ applies. This position is used to calculate the distance to the object.

=== Accumulate method
In this method, the volume densities $σ_i$ accumulate along each ray until the cumulative density reaches or exceeds a predefined threshold $Sigma_(t h)$. For each sampling point $x_i$ along the ray, the cumulative density $S_i =Sigma_(k=1)^i σ_k$ is calculated. The first point $x_(i∗)$ at which $S_i gt.eq Sigma_(t h)$ applies is identified. This position is used to calculate the distance to the object

=== Maximum density change
This method calculates the absolute density changes between successive sample points along each ray. Let $σ_i$ be the volume density at point $x_i$. The density change between points $x_(i-1)$ and $x_i$ is calculated as $Delta_(sigma_i) = abs(sigma_i -sigma_i -1)$. The index $i^∗$ at which the density change $Delta_(sigma_i)$ is maximized is determined: $i^∗ = arg max Delta_(σ_i)$. The corresponding position $x_(i^∗)$ is assumed to be the position where the ray hits the object, and the distance is calculated.

=== Average density change
In comparison to the previous method, where I identified the point along the ray with the maximum density change between consecutive samples, this method similarly computes the absolute density differences between successive points along each ray. However, instead of solely relying on the maximum density change, I calculate the average density difference $accent(Delta_sigma, -)$ across the entire ray. I then define a dynamic threshold $Delta_(t h) = k * accent(Delta_sigma, -) $, where $k$ is a scaling factor.

== Main method
After placing the frustum with the origin and direction, obtaining the vertical and horizontal angular resolution and pressing a button to generate the point cloud within the scene, a Nerfstudio camera object will be created and all densities and locations along each ray will saved in a list.
To remember, the transmittance along a ray is in Nerfstudio defined as: 
$ T(t) = exp(- integral_(t_n)^t sigma (r (s)) d s) $ Where $sigma (r (s))$ is the volume density along the ray depending on the position $r (s)$ on ray $r (t) = o + t d$.
Since the ray is scanned at discrete points, the integral can be approximated by a sum over the segments:\
#align(center,$T_n = product^n_(i=1) exp(-sigma_i ⋅ Delta_(s_i))$) where:\ 
- $sigma_i$ is the density on point $i$ \
- $Delta_(s_i)$ is the distance between point $i-1$ and $i$ \
- $T_o = 1$ the initial transmittance \

A collision is detected when transmittance $T_n$ falls below the threshold: $T_n gt.small T_(t h)$

// #figure(
//   caption: [Pseudocode of main algorithm],
//   pseudocode-list(booktabs: true, title: [Find Collision With Transmittance])[
//     + \""" Finds the collision point along a ray based on transmission values.
//     + ray_densities: 3D coordinates representing points along the ray.
//     + ray_location: density values corresponding to each point.
//     + threshold: The threshold for the transmission probability to consider as a collision
//     + total_density = 0.0
//     + transmittance = 1.0
//     + origin = ray_locations[0]
//     + *For* location, densities
//       + set distance = distance between origin & location
//       + total_distance += distance 
//       + set delta_transmittance = exp(-density \* distance)
//       + transmittance \*= delta_transmittance
//       + *if* transmittance < transmission_threshold
//         + return total_distance, location, density
//     + return None, None, None  
//   ] 
// )


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
== Conclusion <conclusion>
#strong[Vergleich und Bewertung:]

Was sind die genauen Unterschiede zwischen einem echten und einem synthetischen LiDAR in Bezug auf Genauigkeit und Performance?
Wie schneidet dein Modell im Vergleich zu einer realen Simulation ab?
Welche Vor- und Nachteile ergeben sich bei der Nutzung von NeRF für LiDAR-Simulation im Vergleich zu traditionellen Methoden?
Wie steht der Aufwand für die Kalibrierung des synthetischen LiDARs im Vergleich zu einem realen LiDAR?

#strong[Validierung und Tests:]

Wie validierst du die Genauigkeit der von deinem synthetischen LiDAR gesammelten Daten? Gibt es Metriken, die zur Evaluierung verwendet werden?
Welche Tests wurden durchgeführt, um die Robustheit deines synthetischen LiDARs zu überprüfen? Sind die Ergebnisse reproduzierbar?

#strong[Anwendungsbereiche und Limitationen:]

Für welche realen Szenarien wäre die Anwendung deines synthetischen LiDARs in NeRF besonders nützlich?
Gibt es spezielle Szenarien, in denen dein Ansatz nicht funktioniert oder nur eingeschränkt geeignet ist?
Wie würde dein Ansatz in einer dynamischen Szene funktionieren, also wenn sich Objekte im Raum bewegen?


#strong[NVIDIA Omniverse und Integration:]

Wie wird NVIDIA Omniverse in den gesamten Workflow integriert? Ist es möglich, diesen Schritt zu ersetzen oder zu optimieren?
Welche spezifischen Herausforderungen traten bei der Nutzung von NVIDIA Omniverse auf?

#strong[Implementierung und Architektur:]

Welche konkreten Designentscheidungen hast du in Bezug auf die Architektur des NeRF getroffen (z.B. Schichtung, Aktivierungsfunktionen)? Was sind die Gründe hinter diesen Entscheidungen?
Wie genau funktioniert das Frontend in Bezug auf die Nutzerinteraktion? Welche Funktionalitäten sind kritisch für den Nutzer?

#strong[Fazit und Ausblick:]

Welche offenen Fragen oder Herausforderungen gibt es aktuell noch in deinem Ansatz?
Gibt es Möglichkeiten, deinen Ansatz in Zukunft zu erweitern? Zum Beispiel durch bessere Netzwerkarchitekturen oder den Einsatz von neuen Sensormodellen?
#pagebreak()

= Discussion <discussion>
#lorem(450)
#pagebreak()

= Use of AI in this thesis <use-of-ai-in-this-thesis>

#pagebreak()

#bibliography("references.bib", full: false, style:"apa")
