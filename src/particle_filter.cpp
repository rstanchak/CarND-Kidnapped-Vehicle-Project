/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *  Modified on: Aug 8, 2017
 *  	Author: Roman Stanchak
 */

#include <math.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>
#include <vector>

#include "particle_filter.h"

using std::string;
using std::stringstream;
using std::ostream_iterator;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
    std::normal_distribution<double> xdist(x, std[0]);
    std::normal_distribution<double> ydist(y, std[1]);
    std::normal_distribution<double> theta_dist(theta, std[2]);

	num_particles_ = 20;
	for(int i = 0; i < num_particles_; ++i) {
		Particle p = {
			.id = i,
			.x = xdist(rng_),
			.y = ydist(rng_),
			.theta = theta_dist(rng_),
			.weight = 1 };
		particles_.push_back(p);
		weights_.push_back(1);
	}
	is_initialized_ = true;
}

Particle ParticleFilter::predict1(Particle p, double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	double dx = yaw_rate == 0. ? velocity * delta_t * cos(p.theta) : velocity/yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
	double dy = yaw_rate == 0. ? velocity * delta_t * sin(p.theta) : velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
	p.x += dx + std::normal_distribution<>(0, std_pos[0])(rng_);
	p.y += dy + std::normal_distribution<>(0, std_pos[1])(rng_);
	p.theta += yaw_rate * delta_t + std::normal_distribution<>(0, std_pos[2])(rng_);
	return p;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	for(int i = 0; i < num_particles_; ++i) {
		particles_[i] = this->predict1(particles_[i], delta_t, std_pos, velocity, yaw_rate);
	}
}

static double l2sq(double x1, double y1, double x2, double y2) {
	return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
}

static double normal1d(double x, double mu, double sigma) {
	return 1./(sigma*sqrt(2*M_PI))*exp(-(x-mu)*(x-mu)/(2*sigma*sigma));
}

static double normal2d(double x, double y, double mu_x, double std_x, double mu_y, double std_y) {
	return 1./(std_x*std_y*sqrt(2*M_PI))*exp(-(x-mu_x)*(x-mu_x)/(2*std_x*std_x) - (y-mu_y)*(y-mu_y)/(2*std_y*std_y));
}

static int nearestNeighborL2(const std::vector<LandmarkObs> & predicted, LandmarkObs obs) {
	// Find the predicted measurement that is closest to the observed measurement and return the index
	double best_dist = std::numeric_limits<double>::max();
	int best_idx = -1;
	for(int i = 0; i < predicted.size(); ++i) {
		double dist = l2sq(predicted[i].x, predicted[i].y, obs.x, obs.y);
		if(dist < best_dist) {
			best_dist = dist;
			best_idx = i;
		}
	}
	return best_idx;
}

struct MapToVehicleTransform {
	double cos_theta, sin_theta, tx, ty;
	MapToVehicleTransform(double particle_x, double particle_y, double particle_theta):
		cos_theta(cos(particle_theta)),
		sin_theta(sin(particle_theta)),
		tx(-particle_x),
		ty(-particle_y) { }
	void operator()(double x, double y, double & retarg_x, double & retarg_y) {
		retarg_x = (x+tx)*cos_theta + (y+ty)*sin_theta;
		retarg_y = -(x+tx)*sin_theta + (y+ty)*cos_theta;
	}
};

struct VehicleToMapTransform {
	double cos_theta, sin_theta, tx, ty;
	VehicleToMapTransform(double particle_x, double particle_y, double particle_theta):
		cos_theta(cos(-particle_theta)),
		sin_theta(sin(-particle_theta)),
		tx(particle_x),
		ty(particle_y) { }
	void operator()(double x, double y, double & retarg_x, double & retarg_y) {
		retarg_x = x*cos_theta + y*sin_theta + tx;
		retarg_y = -x*sin_theta + y*cos_theta + ty;
	}
};

static void cartesianToPolar2d(double x, double y, double * retarg_rho, double * retarg_phi ) {
	*retarg_rho = sqrt(x*x + y*y);
	*retarg_phi = atan2(y, x);
}

/*
 * Update the weights of a particle using a mult-variate Gaussian distribution.
 * @param sensor_range Range [m] of sensor
 * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
 *   standard deviation of bearing [rad]]
 */
void ParticleFilter::updateWeight(Particle & p, double sensor_range, double std_landmark[],
		          std::vector<LandmarkObs> observations, const Map & map_landmarks) {
	double r2 = sensor_range*sensor_range;
	std::vector<LandmarkObs> predicted;
	VehicleToMapTransform T(p.x, p.y, p.theta);
	std::vector<double> sense_x, sense_y;
	std::vector<int> associations;

	// transform observations in vehicle coordinates to map coordinates
	for(auto iter = observations.begin(); iter != observations.end(); ++iter) {
		double x, y;
		T(iter->x, iter->y, x, y);
		sense_x.push_back(x);
		sense_y.push_back(y);
		iter->x = x;
		iter->y = y;
	}

	// filter the landmarks that are too far from the vehicle
	for(auto iter = map_landmarks.landmark_list.begin(); iter != map_landmarks.landmark_list.end(); ++iter) {
		if(l2sq(iter->x_f, iter->y_f, p.x, p.y) > r2) continue; // landmark is out of sensor range
		predicted.push_back((LandmarkObs){iter->id_i, iter->x_f, iter->y_f});
	}

	p.weight = 1.;

	if(predicted.size() == 0) {
		p.weight = 1e-15;  // no measurements in range; return really small weight to keep things running
		return;
	}

	// find the landmark associated with each observation
	// and update the probability
	MapToVehicleTransform S(p.x, p.y, p.theta);
	for(auto iter = observations.begin(); iter != observations.end(); ++iter) {
		int idx = nearestNeighborL2(predicted, *iter);

		// attribute association
		associations.push_back(predicted[idx].id);

		// transform back to vehicle coordinates and then range, bearing
		double x, y;
		double pred_bearing, pred_range;
		double obs_bearing, obs_range;

		S(predicted[idx].x, predicted[idx].y, x, y);
		cartesianToPolar2d(x, y, &pred_range, &pred_bearing);
		S(iter->x, iter->y, x, y);
		cartesianToPolar2d(x, y, &obs_range, &obs_bearing);

		p.weight *= normal2d(pred_range, pred_bearing, obs_range, std_landmark[0], obs_bearing, std_landmark[1]);
	}
	p = this->SetAssociations(p, associations, sense_x, sense_y);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	for(int i = 0; i < num_particles_; ++i) {
		this->updateWeight(particles_[i], sensor_range, std_landmark, observations, map_landmarks);
		weights_[i] = particles_[i].weight;
	}
}

void ParticleFilter::resample() {
	std::discrete_distribution<> d(weights_.begin(), weights_.end());
	std::vector<Particle> new_particles;
	for(int i = 0; i < num_particles_; ++i) {
		int idx = d(rng_);
		new_particles.push_back(particles_[idx]);
	}
	particles_ = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y) {
	// particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	// Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
