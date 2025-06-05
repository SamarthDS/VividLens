import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Droplets, 
  Sparkles, 
  Video, 
  MessageSquareText,
  ArrowLeft
} from 'lucide-react';

function ModelDetails() {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="bg-navy-950 pt-24">
      {/* Header */}
      <section className="py-16 bg-navy-900">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-3xl mx-auto text-center">
            <h1 className="text-4xl font-bold text-white mb-4">
              Explore Our Advanced AI Models
            </h1>
            <p className="text-xl text-gray-300">
              Discover the Technology Behind VividLens
            </p>
            <Link 
              to="/" 
              className="mt-8 inline-flex items-center text-blue-400 hover:text-blue-300"
            >
              <ArrowLeft className="mr-1 h-4 w-4" /> Back to Home
            </Link>
          </div>
        </div>
      </section>

      {/* Model Cards */}
      <section className="py-16 bg-navy-950">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-6xl mx-auto">
            {/* Remastering Model */}
            <div className="bg-navy-900 rounded-2xl mb-16 overflow-hidden shadow-xl shadow-blue-900/20">
              <div className="grid md:grid-cols-2">
                <div className="order-2 md:order-1 relative aspect-video md:aspect-auto">
                  <img 
                    src="https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" 
                    alt="Remastering example" 
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-l from-navy-900 via-transparent to-transparent md:hidden"></div>
                </div>
                <div className="order-1 md:order-2 p-8 md:p-12 flex flex-col justify-center">
                  <div className="mb-6 text-blue-500">
                    <Sparkles className="h-12 w-12" />
                  </div>
                  <h2 className="text-3xl font-bold text-white mb-4">
                    Video Frame Augmentation Visualizer
                  </h2>
                  <p className="text-gray-300 mb-6 leading-relaxed">
                    The remastering model upscales and enhances blurred or degraded images, ensuring high-quality 
                    visuals for accurate identification and investigation. Using super-resolution algorithms, our 
                    model can significantly improve the clarity of low-resolution footage, making critical details 
                    visible that would otherwise be lost.
                  </p>
                  <div className="space-y-3">
                    <div className="flex items-start">
                      <div className="flex-shrink-0 h-5 w-5 rounded-full bg-blue-500 mt-1"></div>
                      <p className="ml-3 text-gray-300">Up to 8x resolution enhancement</p>
                    </div>
                    <div className="flex items-start">
                      <div className="flex-shrink-0 h-5 w-5 rounded-full bg-blue-500 mt-1"></div>
                      <p className="ml-3 text-gray-300">Noise reduction and artifact removal</p>
                    </div>
                    <div className="flex items-start">
                      <div className="flex-shrink-0 h-5 w-5 rounded-full bg-blue-500 mt-1"></div>
                      <p className="ml-3 text-gray-300">Perfect for recovering details from low-quality footage</p>
                    </div>
                  </div>
                  <a
  href="http://localhost:8502/"
  target="_blank"
  rel="noopener noreferrer"
  className="mt-6 flex justify-center items-center px-6 py-2 bg-blue-800 hover:bg-blue-900 text-white font-semibold rounded-md transition-all duration-300 transform hover:scale-105 shadow-md"
>
  Know More
</a>
                </div>
              </div>
            </div>

            {/* Video Surveillance Anomaly Detection */}
            <div className="bg-navy-900 rounded-2xl mb-16 overflow-hidden shadow-xl shadow-blue-900/20">
              <div className="grid md:grid-cols-2">
                <div className="p-8 md:p-12 flex flex-col justify-center">
                  <div className="mb-6 text-blue-500">
                    <Video className="h-12 w-12" />
                  </div>
                  <h2 className="text-3xl font-bold text-white mb-4">
                    Real-Time Theft Detection
                  </h2>
                  <p className="text-gray-300 mb-6 leading-relaxed">
                    Our anomaly detection model analyzes surveillance footage in real-time, identifying suspicious 
                    activities and alerting authorities for prompt action. The system learns normal patterns of behavior 
                    and can detect deviations that may indicate criminal activity, from unusual movement patterns to 
                    abandoned objects.
                  </p>
                  <div className="space-y-3">
                    <div className="flex items-start">
                      <div className="flex-shrink-0 h-5 w-5 rounded-full bg-blue-500 mt-1"></div>
                      <p className="ml-3 text-gray-300">Real-time monitoring and alerts</p>
                    </div>
                    <div className="flex items-start">
                      <div className="flex-shrink-0 h-5 w-5 rounded-full bg-blue-500 mt-1"></div>
                      <p className="ml-3 text-gray-300">Low false-positive rate through advanced algorithms</p>
                    </div>
                    <div className="flex items-start">
                      <div className="flex-shrink-0 h-5 w-5 rounded-full bg-blue-500 mt-1"></div>
                      <p className="ml-3 text-gray-300">Customizable sensitivity based on environment</p>
                    </div>
                  </div>
                  <a
  href="http://localhost:8503/"  
  target="_blank"
  rel="noopener noreferrer"
  className="mt-6 flex justify-center items-center px-6 py-2 bg-blue-800 hover:bg-blue-900 text-white font-semibold rounded-md transition-all duration-300 transform hover:scale-105 shadow-md"
>
  Know More
</a>
                </div>
                <div className="relative aspect-video md:aspect-auto">
                  <img 
                    src="https://images.pexels.com/photos/2883120/pexels-photo-2883120.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" 
                    alt="Anomaly detection example" 
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-r from-navy-900 via-transparent to-transparent md:hidden"></div>
                </div>
              </div>
            </div>

            {/* Image Captioning System */}
            <div className="bg-navy-900 rounded-2xl mb-16 overflow-hidden shadow-xl shadow-blue-900/20">
              <div className="grid md:grid-cols-2">
                <div className="order-2 md:order-1 relative aspect-video md:aspect-auto">
                  <img 
                    src="https://images.pexels.com/photos/5473302/pexels-photo-5473302.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" 
                    alt="Image captioning example" 
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-l from-navy-900 via-transparent to-transparent md:hidden"></div>
                </div>
                <div className="order-1 md:order-2 p-8 md:p-12 flex flex-col justify-center">
                  <div className="mb-6 text-blue-500">
                    <MessageSquareText className="h-12 w-12" />
                  </div>
                  <h2 className="text-3xl font-bold text-white mb-4">
                    Surveillance Video Analysis
                  </h2>
                  <p className="text-gray-300 mb-6 leading-relaxed">
                    The image captioning system automatically generates descriptive captions for images, aiding in the 
                    quick and accurate interpretation of visual data. This powerful tool can process thousands of images 
                    rapidly, creating searchable text descriptions that highlight key elements in each frame, from people 
                    and objects to activities and environmental conditions.
                  </p>
                  <div className="space-y-3">
                    <div className="flex items-start">
                      <div className="flex-shrink-0 h-5 w-5 rounded-full bg-blue-500 mt-1"></div>
                      <p className="ml-3 text-gray-300">Detailed scene descriptions in natural language</p>
                    </div>
                    <div className="flex items-start">
                      <div className="flex-shrink-0 h-5 w-5 rounded-full bg-blue-500 mt-1"></div>
                      <p className="ml-3 text-gray-300">Object and action recognition with high accuracy</p>
                    </div>
                    <div className="flex items-start">
                      <div className="flex-shrink-0 h-5 w-5 rounded-full bg-blue-500 mt-1"></div>
                      <p className="ml-3 text-gray-300">Creates searchable database of visual content</p>
                    </div>
                  </div>
                  <a
  href="http://localhost:8501/"
  target="_blank"
  rel="noopener noreferrer"
  className="mt-6 flex justify-center items-center px-6 py-2 bg-blue-800 hover:bg-blue-900 text-white font-semibold rounded-md transition-all duration-300 transform hover:scale-105 shadow-md"
>
  Know More
</a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 bg-gradient-to-b from-navy-900 to-navy-950">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
         
        </div>
      </section>
    </div>
  );
}

export default ModelDetails;