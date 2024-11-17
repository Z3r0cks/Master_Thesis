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
#set heading(numbering: "I.")

= Introduction <introduction>
Sensors enable machines to perceive and interpret the world around them. By allowing humans, through machines, to observe and measure aspects of the environment beyond human capability, sensors foster the development of technologies and machines that enhance our understanding of the world, physics, human behavior, and more. They are, therefore, indispensable for technological advancement.

This thesis explores whether a synthetic LiDAR sensor can be generated within a Neural Radiance Field (NeRF) framework. By simulating LiDAR sensors in a NeRF, the aim is to determine the feasibility of virtual sensor models, offering insights into which physical sensors may be suitable or necessary for real-world applications. It explores the advantages and disadvantages of this approach, the requirements and differences compared to the data from a simulated LiDAR sensor.

Industrial applications can be dangerous. The use of inappropriate sensors can cause injuries or worse. Testing potential sensors in a virtual environment, such as a NeRF, is safer and more efficient than experimenting with real sensors. Therefore, developing a method to simulate LiDAR sensors within NeRFs could significantly aid in selecting the correct sensors for specific applications. To understand how a LiDAR sensor can be simulated within a NeRF, it is essential to delve into the principles of distance measurement and the role of constants like the velocity of light. 

An important area for sensors is measuring, especially distance measuring. In order to facilitate measurement, it is necessary to employ a constant value that is universally applicable and accessible to all. One such fundamental physical constant is the velocity of light, which can be used to calculate the distance between two points in space by determining the time required for light to travel between them. This concept forms the basis of the measurement principle known as "time-of-flight" (ToF) #cite(<hahne_real-time_2012>, supplement: [p.~11]). One prominent example of a time-of-flight sensor is Light Detection and Ranging (LiDAR). They are different types of LiDAR. One type of LiDAR-the focus of this thesis-is the use of lasers to aim at an object and then measure the time. it takes for the reflected light to return to the receiver, allowing it to generate a 3D or 2D image with spatial and depth data @noauthor_sick-lidar_nodate. It is difficult to estimate which LiDAR could be correct for each given application, highlighting the need for effective simulation methods. By focusing on simulating LiDAR sensors within a NeRF, this thesis aims to provide a tool for assessing the suitability of various LiDAR technologies in a safe and controlled virtual environment.

This approach could offer significant advantages in terms of safety, cost, and efficiency when determining the most appropriate sensor for specific industrial applications. To effectively simulate LiDAR sensors in a virtual environment, advanced methods for 3D scene representation are required.

Measuring the real world is difficulty. We can use cameras but those are always 2D scene where measuring is limited. The use of an LiDAR sensor to measure the real world, resolves in a 3D point cloud, which is difficulty to understand. One such methods to render a 3D scene from 2D images is the Neural Radiance Fields (NeRF). 

In 2020, #cite(<mildenhall_nerf_2020>, form: "author") introduced NeRF as an AI-based approach for generating 3D representations of scenes from 2D images. One of the key advantages by using NeRF is the low memory usage and the photo-realistic representation. With the use of pipelines, it is also easy to use NeRFs, even for untrained users @schleise_automated_2024. NeRF can be extended with other methods, such as object recognition through integrated additional neural networks or scene manipulation. The central idea behind NeRF is to utilize a neural network, rather than techniques that use a grid or a list system to determinate each point and its corresponding values within a 3D scene. However one problems which NeRF also address are the ill-posed problem:

- #strong[Non-unique solution:] There is no clear way to reconstruct a 3D scene from limited 2D images.

- #strong[Sensitivity:] Small changes in the input can have big changes in the output.

- #strong[Incompleteness of the data:] Not all information about the scene is known in the input data.

NeRF address this ill-posed problem by utilizing a neural network that learns contextual information from the scene. The neural network minimizes the loss function between its computed outputs and the ground truth images, thereby learning to approximate the original scene. This approach enables reconstruction even from perspectives for which no images are provided, as long as information is available from other images.
\
Like other neural networks, NeRF suffers from the disadvantage of being a \"black box\" @la_rocca_opening_2022. This makes it difficult or even impossible to fully understand the computations taking place, as the neural network provides limited interpretability. As a result, it is challenging to determine positions, distances, or lengths within the scene. Unlike methods, which typically store scene data in grids or lists, where each pixel within the scene is explicit know, NeRF only compute RGB and density values from given coordinates without reference to the original scene.\

#figure(image("images/introduction/implicite.png", width: 100%), caption: [Comparison between implicit and explicit coordinates in NVIDIA Omniverse and a NeRF scene representation. The use of NVIDIA Omniverse to calculate the distance between two known points in a 3D scene using the Pythagorean theorem (A). The coordinates as user input: $x'_a , y'_a , z'_a$ for the neural network $f_theta$ which interprets these coordinates by computing pixels in the space it learns from training (B). Illustration of the problem of getting a point closest to an obstacle in NeRF (C).#text(fill:blue, [Ich suche immer noch nach einer besseren Darstellung für das rechte Bild. Es wird nicht ganz klar, was das Problem darstellt.])])<fig:impicit>

As illustrated in @fig:impicit it is challenging to obtain the requisite coordination to measure the distance between the two points. While the distance can be readily calculated in NVIDIA Omniverse due to the availability of known coordinates, the coordinates in a NeRF scene are dependent on the original images and their associated parameters. It is feasible to approximate the same coordinates as in NVIDIA Omniverse. However, due to the issue of infinite points in space and the lack of knowledge regarding the coordinate of each object within the scene, it is not possible to obtain the exact coordinates, which represents a significant challenge in the implementation of this approach with LiDAR sensors, which are highly precise. Nevertheless, even when it would be feasible to obtain the exact positions, the distance would not be accurate, given that the NeRF scene lacks any references regarding scale. This understanding also depends on the camera parameters. Another Problem is to obtain the closest point on an obstacle. A NeRF approximates the original scene by approximates the RGB and density values. These values are not exact representations of the scene. In the real world or in graphical software, it is possible to use a pen or click on an object because its coordinates are known and defined. However, in a NeRF, every pixel is only an approximation of an RGB value, influenced by the density value depending from images. This means that it is not possible to take a pen and mark a specific point, because the exact position and properties of that point are not explicitly defined in the model. \

The employment of multiple lasers facilitates the generation of a point cloud representation of the surrounding environment. These sensors are specifically designed for real-time applications. However, the utilization of a NeRF to synthesize data in real-time remains a challenge. The computation of distance between two points within a NeRF scene an generating a point cloud necessitates the completion of several steps.
#v(2mm) 
#strong[Determine the points:] This represents a significant challenge with regard to implementation. In other graphical software, such as Unity, NVIDIA Omniverse, and Blender, the coordinates of each edge, vertex, or face are explicitly defined. However, in NeRF, these coordinates are not known. To illustrate this concept in a simplified manner, each scene in NeRF can be conceptualized as a "memory" of the neural network, which learns to generate RGB and density values from a multitude of perspectives and positions by minimizing the discrepancy between the original image and its internal representation. This is referred to as the loss function. This discrepancy in scale and coordinates between the original scene and the NeRF scene presents a challenge. Although this implementation assumed that the local origin in the global scene could be set manually with a visible object within the scene that is movable in runtime, the primary issue lies in determining the appropriate distance between this local origin and each potential obstacle that a ray might encounter. As in volume sampling described, the density value increases close to an obstacle. To address this issue, multiple methods are developed and tested to estimate the sample point closest to the obstacle by recognize the increasing density value.
#v(2mm) 
#strong[Casting rays:] Once the origin within the scene has been established, rays must be cast in order to obtain the requisite distance measurement and point cloud reflection. The rays should be dependent on the local origin and not on the scene perspective, which are typically employed in NeRF for scene representation.
#v(2mm) 
#strong[Distance measurement:] To measure the distance between two points, the Pythagorean theorem is applied. A key challenge, however, lies in the fact that the scale within the scene does not correspond accurately to that of the original scene, particularly when camera parameters are estimated rather than precisely known during training.
#v(2mm) 
#figure(image("images/related_work/lidar_vs_nerf.png"), caption: [Simplified illustration depicts the concept of LiDAR and the rationale behind this thesis. While a time-of-flight sensor allows light to pass through and measures the time it takes for light to travel to an obstacle and reflect back to a receiver (illustrated on the left), a NeRF does not actually cast rays within the scene. The challenge is to identify the sample point that is closest to the obstacle and compute the distance between the origin and the sample point, as the coordinates of the origin and the sample point are within the scene.])
\
The implementation can be employed for the purpose of measuring distances within a given scene, in a manner analogous to that of an artificial LiDAR. Additionally, it can be utilized for the precise measurement of distances between two points within the scene. Furthermore, the implementation allows for the use of a reference object within the scene, which is used to compute a scale factor that is then applied to the measured distances. \
In order to implement the method, a technique is employed which enables the detection of the density value at a given point within the scene, as well as the calculation of the distance between the origin and the points. NVIDIA Omniverse is utilized for the generation of images, the management of camera parameters and as a means of comparison with a simulated LiDAR. Meanwhile, Nerfstudio with Nerfacto is employed for the creation of the NeRF and the implementation of the proposed methods.



#pagebreak()

#bibliography("references.bib", full: false, style:"apa")