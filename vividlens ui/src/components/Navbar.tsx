import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Eye } from 'lucide-react';

const Navbar: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 10) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <header 
      className={`fixed w-full z-30 transition-all duration-300 ${
        isScrolled ? 'bg-navy-950 shadow-lg' : 'bg-transparent'
      }`}
    >
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          <Link to="/" className="flex items-center space-x-2">
            <Eye className="h-8 w-8 text-blue-500" />
            <span className="text-2xl font-bold text-white">VividLens</span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex space-x-10">
            <Link 
              to="/" 
              className={`text-sm font-medium transition-colors ${
                location.pathname === '/' 
                  ? 'text-blue-400' 
                  : 'text-gray-200 hover:text-blue-400'
              }`}
            >
              Home
            </Link>
            <Link 
              to="/models" 
              className={`text-sm font-medium transition-colors ${
                location.pathname === '/models' 
                  ? 'text-blue-400' 
                  : 'text-gray-200 hover:text-blue-400'
              }`}
            >
              Our Models
            </Link>
       
          </nav>

          <div className="hidden md:block">
            <Link 
              to="#" 
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 transition-colors"
            >
              Get Started
            </Link>
          </div>

          {/* Mobile Navigation Button */}
          <div className="md:hidden">
            <button
              type="button"
              className="text-gray-200 hover:text-white"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              {isMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation Menu */}
      {isMenuOpen && (
        <div className="md:hidden bg-navy-900">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            <Link
              to="/"
              className={`block px-3 py-2 rounded-md text-base font-medium ${
                location.pathname === '/'
                  ? 'text-white bg-blue-600'
                  : 'text-gray-200 hover:bg-blue-700 hover:text-white'
              }`}
              onClick={() => setIsMenuOpen(false)}
            >
              Home
            </Link>
            <Link
              to="/models"
              className={`block px-3 py-2 rounded-md text-base font-medium ${
                location.pathname === '/models'
                  ? 'text-white bg-blue-600'
                  : 'text-gray-200 hover:bg-blue-700 hover:text-white'
              }`}
              onClick={() => setIsMenuOpen(false)}
            >
              Our Models
            </Link>
            <a
              href="#"
              className="block px-3 py-2 rounded-md text-base font-medium text-gray-200 hover:bg-blue-700 hover:text-white"
              onClick={() => setIsMenuOpen(false)}
            >
              About
            </a>
            <a
              href="#"
              className="block px-3 py-2 rounded-md text-base font-medium text-gray-200 hover:bg-blue-700 hover:text-white"
              onClick={() => setIsMenuOpen(false)}
            >
              Contact
            </a>
            <Link
              to="#"
              className="block px-3 py-2 rounded-md text-base font-medium text-white bg-blue-600 hover:bg-blue-700"
              onClick={() => setIsMenuOpen(false)}
            >
              Get Started
            </Link>
          </div>
        </div>
      )}
    </header>
  );
};

export default Navbar;