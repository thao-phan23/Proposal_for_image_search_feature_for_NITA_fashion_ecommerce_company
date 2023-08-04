# Project proposal: SHOP BY IMAGE SEARCH
![image](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/b3ec5d8a-3fa3-4307-8602-9ade36eed20d)
# 1. Overview:

### About the company: 
NITA Fashion is one of the biggest e-commerce companies distributing over 400 local and international brands. The products are divided into four main categories: apparel, footwear, accessories, and personal care. Customer experience is the company's core value. We consistently strive to identify challenges and devise effective solutions, placing utmost importance on ensuring our customers' satisfaction.

### Business problem:
As an E-commerce data scientist, I am recently working on a project proposal to enhance the customer experience during product searches. This initiative is based on insights gathered from a survey conducted in 2022, which involved feedback from 2000 customers regarding the challenges they encounter while searching for products using keywords. The survey findings revealed several valuable points:
- 35% of the customers reported difficulties in locating their desired products amidst the vast array of offerings on the website.
- 42% expressed concerns about spending excessive time on the website without being able to find their target items effectively.
- 22% of the customers expressed the demand for an image search feature

### Key business question: 
How to improve customer experience during product searches?

### Visual search:

I would like to mention visual search (image search and video search) and how potential it is, especially in the fashion e-commerce market.
- Visual search is a new search type with rising demand, in which:
  - 62% of Millennial and Generation Z customers desire visual search over any other new technology. (2019)
  - 54% of US internet users were excited to have this technology in their shopping experience
    ![vs](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/9c6a4538-94a8-45b9-b380-87a8aabf7935)
    
  - Fashion has been always at the top of visual search
    
    ![fs](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/d7afa89e-2cea-4c73-ba4b-fd4cbbba9ee5)

  - The visual search market is projected to experience substantial growth, with a Compound Annual Growth Rate (CAGR) of approximately 18% anticipated until the year 2028.
    
    ![cagr](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/f8643baa-d004-455b-9ecf-c463bf71da03)

Considering the vast potential of the global visual search market, I believe that adopting visual search is a critical and timely move for our company. Being an early adopter will provide us with competitive advantages and allow us to leverage all of its benefits.

### Solution:
Given our company's current capacity and available resources, I propose taking a phased approach, starting with phase 1, which will primarily focus on building an image search feature with 2 functions: image classification and image similarity. This will allow us to build a strong foundation and gradually expand into other areas of visual search in the future.

### Stakeholders:
The proposal will be introduced to the Chief Executive Officer and all Department Heads to get feedback and approval for its implementation.

# 2. Data understanding:
![image](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/9a6822de-bdab-4cb6-aa72-421f51733b83)

Our dataset consists of more than 44 thousand images from 143 product types, however, for proposal purposes, I will focus on the top 10 product types with more than 24 thousand images.

# 3. Modeling
### 3.1. Image classification modeling:
We will utilize both classic convolutional neural networks and transfer learning networks to build up the first foundational model for image classification tasks, then compare their performance and select the best one.

#### Convolutional neural networks (CNN):
- The best CNN model achieved a test accuracy of 95%.

  ![best cnn](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/00996d42-b8b4-4ec0-957a-9b12c2197ca1)

#### Transfer learning networks:
We then apply transfer learning networks to build up the models to see which is the best one to utilize.

- The best ResNet50 model has an accuracy of 75%.

    ![resnet50](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/3288828a-104f-4265-a7af-dd3285e5eb3d)

- MobiletNetV2 best model improves the test accuracy to 79%.

    ![mobilenet](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/ffefcb86-9126-4dc5-a870-e2f83f615d6b)

- VGG16 has an impressive test accuracy of 95%.

    ![vgg16](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/bb88caf9-1560-4e16-b95b-0656cdf5200c)


### 3.2. Image similarity modeling:

We will build up a Siamese model with utilizing contrastive loss in order to predict the Euclidean distances of image pairs. The shorter distances indicate stronger similarity.

  ![siamese](https://github.com/thao2023/Image_search_feature_proposal_for_fashion_website/assets/131706716/d54bcb03-a35c-4fdc-81bd-dc65a04fa63f)

Test accuracy first achieved 50% and despite various improvement steps, the model's test accuracy remain at only 50%. Due to the time constraint and computational limitations, we will consider this model is our best model at the moment.

# 4. Model evaluation:


