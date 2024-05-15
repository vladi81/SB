# Identification of seals and dolphins from the beach cameras

Unlike people swimming in the sea, seals and dolphins coming close to a beach is a rare phenomenon. Collecting data with beach cameras to build detection models can be very long and not cost effective process. I see few alternatives to that:
1. Using data from Internet
2. Collecting data in dolphin/seal marine parks
3. Simulating dolphins/seals in the sea


## 1. Data from Internet
Dolphins are quite a popular sea animal. There are videos from various locations, recorded by amateur and professional photographers and shared on Youtube:
- [Dolphin Days (Full Show) at SeaWorld San Diego on 8/30/15](https://www.youtube.com/watch?v=vhGHgIkV3J4)
- [Incredible Dolphin Moments | BBC Earth](https://www.youtube.com/watch?v=G7L4YzGAvMA)
- [A Showy Dolphin Super-Pod | Destination WILD](https://www.youtube.com/watch?v=IY7g1JCfRgk)

There are also popular movies featuring dolphins such as Free Willy, Dolphin Tale, Dolphin Reef.
Nature movies may contain dolphins in the sea captured from a beach, but also a lot of irrelevant data that need to be filtured out such as underwater photography of dolphins, high zoom on dolphins, areal photography not representative of beach camera viewpoints.
Dolphin shows are not in the open sea but it is in the outdoor environment from a viewpoint similar to a beach camera. 

Seal videos can also be found on Youtube. For example:
- [Taronga Zoo Seal Show](https://www.youtube.com/watch?v=_bqXcCkd5oE)
- [Sea Lions Tonite 2015 (Full Show) at SeaWorld San Diego](https://www.youtube.com/watch?v=fQ8Gn0vgG_s)
- [Seals & Pelicans at Pier 39 in San Francisco](https://www.youtube.com/watch?v=HY248sWH-ZU)  


The variety of footage on Internet, however, may be limited. For example, I found some promo videos for Dolphin Reef in Eilat but not "a day from the life of dolphins in Eilat". In addition, dolphin/seal species relevant to a customer in the southern hemisphere may differ in their appearance than footage of common dolphin/seals on Internet.


## 2. Data from dolphin/seal marine parks  
Some of limitations of the public data may be overcome by collecting video footage directly in marine parks. There are marine parks with seals and dolphins in various locations, such as:
- Dolphin Reef in Eilat (Dolphins)
- Adelaide Dolphin Sanctuary in Australia (Dolphins)
- Alonissos Northern Sporades in Greece (Seals)  

Perhaps one could partner with some of the marine parks to install a camera, or even better if in some parks cameras are already installed, and permission to get access to cameras could be received. The footage could improve data quality by having viewpoints representative of beach cameras, covering natural swimming behaviour of dolphins/seals, longer sequences, and variety of weather conditions.
It may still be limited to cover the appearance of common dolphin/seal species rather than dolphin/seal species specific to the southern hemisphere.


## 3. Simulating dolphins/seals in the sea
Another way to collect data for dolphin/seal detection models is to simulate the appearance of dolphins/seals in the sea. This can be done via realistic dolphin/seal 3D models. Here are few examples of such models:  
- [Ocean Park High Quality and Highly Simulated Animatronic Dolphin Model Hot Sale for Adventure Park](https://www.alibaba.com/product-detail/Ocean-Park-High-Quality-and-Highly_1601026071334.html?spm=a2700.galleryofferlist.normal_offer.d_image.52eb67cepgiyOd)
- [High quality dinosaur park simulation animal sea lion animatronic model](https://www.alibaba.com/product-detail/High-quality-dinosaur-park-simulation-animal_1600561903911.html?spm=a2700.galleryofferlist.normal_offer.d_image.36076fedy1Gyx7)

Dolphin/seal 3D model with appearance resembling species relevant to the southern hemisphere can be customly manufactured, added weights below if needed, and dragged in the sea by a boat/speedboat in front of beach cameras of the customer. 
Arguably, this is most expensive way to get data but also most convincing to demonstrate the ability to detect dolphins/seals.
Simulating dolphins/seals in the sea may also have some limitations in capturing not only the appearance but also the behaviour of real dolphins/seals like jumping out of water.  


## Southern Hemisphere Seals and Dolphins
There are species of seals and dolphins in the southern hemisphere with a distict appearance. For example, 
[Southern right whale dolphin](https://en.wikipedia.org/wiki/Southern_right_whale_dolphin) does not have a dorsal fin and has a different pigmentation pattern than [Common bottlenose dolphin](https://en.wikipedia.org/wiki/Common_bottlenose_dolphin) typically found in marine parks.

Similarly, [Southern Elephant Seal](https://en.wikipedia.org/wiki/Southern_elephant_seal) has different pigmentation pattern than [California sea lion](https://en.wikipedia.org/wiki/California_sea_lion) which is a popular choice for marine parks.



## ML Process Design
Looking for relevant data on Internet is the easiest way to start and allows to build a demo detector.  
Collecting data in marine parks comes next, and allows to build general dolphin/seal detector of production quality.  
Simulating dolphins and seals in the sea allows fine tuning of general-purpose dolphin/seal detection models to appearances of species in the southern hemisphere and is good for testing the models in conditions as close as possible to real.
