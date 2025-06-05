import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Droplets, 
  Sparkles, 
  Video, 
  MessageSquareText, 
  ArrowRight 
} from 'lucide-react';

function Home() {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="overflow-x-hidden">
      {/* Hero Section */}
      <section className="pt-24 pb-16 md:pt-32 md:pb-24 relative bg-navy-950">
        <div className="absolute inset-0 z-0 overflow-hidden">
          <div 
            className="absolute inset-0 bg-gradient-to-r from-navy-950/90 to-navy-950/80 z-10"
          ></div>
          <video 
            className="w-full h-full object-cover"
            autoPlay 
            muted 
            loop 
            playsInline
          >
            <source 
              src="https://assets.mixkit.co/videos/preview/mixkit-security-surveillance-camera-view-5157-large.mp4" 
              type="video/mp4" 
            />
          </video>
        </div>

        <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white leading-tight mb-6 animate-fadeIn">
              Revolutionize Crime Surveillance with 
              <span className="text-blue-500 ml-2">VividLens</span>
            </h1>
            <p className="text-xl text-gray-300 mb-8 animate-fadeIn animation-delay-200">
              Empowering Law Enforcement with Advanced AI Technology
            </p>
            <div className="flex flex-col sm:flex-row justify-center gap-4 animate-fadeIn animation-delay-400">
              <Link
                to="/models"
                className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-md transition-all duration-300 transform hover:scale-105"
              >
                Learn More
              </Link>
            </div>
          </div>
        </div>

        <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-navy-900 to-transparent"></div>
      </section>

      {/* Introduction */}
      <section className="py-16 bg-navy-900">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-3xl font-bold text-white mb-6">
              Enhanced Surveillance for a Safer Tomorrow
            </h2>
            <p className="text-gray-300 text-lg leading-relaxed">
              VividLens combines cutting-edge AI technologies to transform the capabilities of law enforcement agencies. 
              Our comprehensive suite of machine learning models is designed to enhance image quality, 
              detect anomalies, and provide valuable insights from surveillance footage.
            </p>
          </div>
        </div>
      </section>

      {/* Features Overview */}
      <section className="py-16 bg-navy-950">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white mb-4">Advanced AI Features</h2>
            <p className="text-gray-300 max-w-3xl mx-auto">
              Our integrated suite of machine learning models provides unprecedented capabilities 
              for crime detection and investigation.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 justify-center mx-auto w-fit">
            {/* Remastering Model */}
            <div className="bg-navy-800 rounded-xl p-6 transition-transform duration-300 hover:transform hover:scale-105">
              <div className="mb-4 text-blue-500">
                <Sparkles className="h-12 w-12" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-3">Video Frame Augmentation Visualizer</h3>
              <p className="text-gray-300">
                Upscale and enhance blurred or degraded images for superior quality.
              </p>
              <Link 
                to="/models" 
                className="mt-4 inline-flex items-center text-blue-400 hover:text-blue-300"
              >
                Learn more <ArrowRight className="ml-1 h-4 w-4" />
              </Link>
            </div>

            {/* Video Surveillance */}
            <div className="bg-navy-800 rounded-xl p-6 transition-transform duration-300 hover:transform hover:scale-105">
              <div className="mb-4 text-blue-500">
                <Video className="h-12 w-12" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-3">Real-Time Theft Detection</h3>
              <p className="text-gray-300">
                Real-time detection of suspicious activities in surveillance footage.
              </p>
              <Link 
                to="/models" 
                className="mt-4 inline-flex items-center text-blue-400 hover:text-blue-300"
              >
                Learn more <ArrowRight className="ml-1 h-4 w-4" />
              </Link>
            </div>

            {/* Image Captioning */}
            <div className="bg-navy-800 rounded-xl p-6 transition-transform duration-300 hover:transform hover:scale-105">
              <div className="mb-4 text-blue-500">
                <MessageSquareText className="h-12 w-12" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-3">Surveillance Video Analysis</h3>
              <p className="text-gray-300">
                Automatically generate descriptive captions for images to aid in investigations.
              </p>
              <Link 
                to="/models" 
                className="mt-4 inline-flex items-center text-blue-400 hover:text-blue-300"
              >
                Learn more <ArrowRight className="ml-1 h-4 w-4" />
              </Link>
            </div>
          </div>
        </div>
      </section>


      {/* CTA Section */}
      <section className="py-16 bg-gradient-to-r from-blue-900 to-navy-900">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-3xl font-bold text-white mb-6">
              Ready to Transform Your Surveillance Capabilities?
            </h2>
            <p className="text-xl text-gray-300 mb-8">
              Join hundreds of law enforcement agencies that trust VividLens for advanced crime detection and investigation.
            </p>
            <div className="flex flex-col sm:flex-row justify-center gap-4">
              <Link
                to="/models"
                className="px-8 py-3 bg-transparent border border-white hover:bg-white/10 text-white font-medium rounded-md transition-all duration-300 transform hover:scale-105"
              >
                Explore Our Tech
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Home;