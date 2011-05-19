/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 TELEMATICS LAB, DEE - Politecnico di Bari
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Giuseppe Piro  <g.piro@poliba.it>
 */


#ifndef MULTIPATH_V30_M10_H_
#define MULTIPATH_V30_M10_H_

static double multipath_M10_v_30[3000] = { 
  1.08872, 1.02776, 0.911288, 0.697174, 0.157452, 0.332127, 0.736656, 0.913191, 1.00548, 1.04635, 1.04688, 1.00936, 0.929495, 0.792641, 0.554533, -0.0326824, 0.176025, 0.53964, 0.680857, 0.732331, 0.721307, 0.649297, 0.494959, 0.162251, -0.467763, 0.30844, 0.535401, 0.641113, 0.679176, 0.663495, 0.59226, 0.445497, 0.14398, -0.994694, 0.17944, 0.426885, 0.538178, 0.576631, 0.557292, 0.476481, 0.306, -0.0863809, -0.335999, 0.220474, 0.420966, 0.514805, 0.543954, 0.518374, 0.432182, 0.254805, -0.158264, -0.330489, 0.188944, 0.38196, 0.471329, 0.496376, 0.465804, 0.371838, 0.177842, -0.312295, -0.240498, 0.190079, 0.365066, 0.445542, 0.464048, 0.426821, 0.32347, 0.10946, -0.499404, -0.164935, 0.20232, 0.362142, 0.435695, 0.450289, 0.410793, 0.306686, 0.0951413, -0.481442, -0.218689, 0.15929, 0.318822, 0.388948, 0.397145, 0.34629, 0.219253, -0.0573613, -1.57807, -0.0258081, 0.246014, 0.378264, 0.439699, 0.449124, 0.409852, 0.313717, 0.130727, -0.268778, -0.563772, -0.00953234, 0.171794, 0.237107, 0.22174, 0.116136, -0.164408, -0.886373, 0.0193766, 0.291915, 0.441734, 0.528375, 0.570951, 0.576686, 0.547028, 0.478632, 0.36107, 0.16662, -0.207542, -0.820844, -0.117091, 0.063878, 0.115932, 0.082747, -0.0428514, -0.339724, -1.2645, -0.267951, -0.0398562, 0.0421682, 0.0263866, -0.108051, -0.564402, -0.390655, 0.0717636, 0.289871, 0.416974, 0.487134, 0.510914, 0.488193, 0.40752, 0.229677, -0.254108, -0.109709, 0.341354, 0.553303, 0.681115, 0.760286, 0.804247, 0.818795, 0.80589, 0.764643, 0.690812, 0.57415, 0.388901, 0.0435252, -0.751592, 0.118239, 0.349794, 0.464467, 0.522651, 0.54485, 0.540779, 0.516138, 0.474915, 0.420288, 0.354803, 0.279867, 0.194362, 0.0917775, -0.0465148, -0.276969, -0.996575, -0.396204, -0.00795666, 0.210886, 0.361197, 0.469231, 0.544848, 0.591836, 0.610641, 0.598743, 0.549315, 0.446238, 0.243722, -0.317806, -0.032397, 0.372664, 0.566062, 0.67843, 0.740798, 0.763675, 0.749121, 0.692532, 0.578526, 0.359489, -0.258753, 0.102958, 0.47717, 0.651618, 0.746116, 0.789662, 0.791321, 0.750926, 0.658933, 0.485021, 0.100753, -0.184131, 0.400032, 0.616823, 0.731019, 0.787722, 0.80122, 0.774904, 0.70451, 0.57407, 0.33236, -0.373326, 0.0997598, 0.436607, 0.586816, 0.6575, 0.67444, 0.643392, 0.557609, 0.388698, 0.0127155, -0.299933, 0.300873, 0.52208, 0.640501, 0.702939, 0.725181, 0.712959, 0.66662, 0.581387, 0.443705, 0.215187, -0.287868, -0.302771, 0.119056, 0.273852, 0.334229, 0.333813, 0.278883, 0.158509, -0.0745336, -0.746368, -0.322918, 0.0237768, 0.179136, 0.255829, 0.281938, 0.266251, 0.208203, 0.0970093, -0.102663, -0.560683, -0.601532, -0.12711, 0.0710656, 0.180773, 0.240578, 0.263583, 0.253394, 0.206955, 0.111651, -0.0709211, -0.532139, -0.451824, 0.027354, 0.25734, 0.405796, 0.509648, 0.582803, 0.631516, 0.658549, 0.664561, 0.648467, 0.607099, 0.533847, 0.414593, 0.21276, -0.233516, -0.323571, 0.167071, 0.366926, 0.475562, 0.532707, 0.552223, 0.539249, 0.494345, 0.414167, 0.289772, 0.099803, -0.220807, -1.31468, -0.461219, -0.2608, -0.256909, -0.418404, -1.10209, -0.540191, -0.191855, -0.0430759, 0.00251323, -0.049736, -0.268798, -1.32714, -0.0837375, 0.236251, 0.422604, 0.542091, 0.615361, 0.649867, 0.646365, 0.599201, 0.490946, 0.265727, -0.519754, 0.147152, 0.494459, 0.671453, 0.775222, 0.83173, 0.850034, 0.831685, 0.771969, 0.655796, 0.4372, -0.154648, 0.145606, 0.52728, 0.701051, 0.793389, 0.834269, 0.833068, 0.789798, 0.695135, 0.519343, 0.137823, -0.181511, 0.414935, 0.630916, 0.743429, 0.798863, 0.812394, 0.788797, 0.726706, 0.61771, 0.438348, 0.1071, -1.04938, 0.0886269, 0.307277, 0.387054, 0.387991, 0.318427, 0.155638, -0.222839, -0.533928, 0.0491148, 0.249472, 0.339041, 0.361514, 0.327176, 0.230219, 0.040706, -0.37745, -0.637919, -0.126276, 0.021049, 0.0347241, -0.0788426, -0.483294, -0.434747, 0.0637654, 0.285128, 0.407862, 0.468211, 0.475688, 0.426062, 0.295061, -0.0191258, -0.54493, 0.210364, 0.463193, 0.599364, 0.671203, 0.693639, 0.66793, 0.582935, 0.398734, -0.102504, 0.0666914, 0.502395, 0.70375, 0.819792, 0.884451, 0.90957, 0.898511, 0.848847, 0.750463, 0.574796, 0.2101, -0.24646, 0.41471, 0.63112, 0.735934, 0.77793, 0.770598, 0.7138, 0.593797, 0.36186, -0.308092, 0.111832, 0.455256, 0.602931, 0.665407, 0.666616, 0.606585, 0.461393, 0.120046, -0.290278, 0.380302, 0.616908, 0.743245, 0.808625, 0.828149, 0.804889, 0.732856, 0.590237, 0.297055, -0.831705, 0.360801, 0.629618, 0.768556, 0.843483, 0.874286, 0.86772, 0.823749, 0.735479, 0.582495, 0.295985, -1.16826, 0.239894, 0.517629, 0.652025, 0.720232, 0.745022, 0.735171, 0.693512, 0.618911, 0.505719, 0.339848, 0.0827565, -0.446867, -0.505235, -0.112798, 0.00877293, 0.0361813, 0.00713566, -0.062698, -0.160563, -0.266007, -0.343492, -0.355646, -0.301237, -0.217057, -0.140185, -0.0931086, -0.0914419, -0.157591, -0.356626, -1.42339, -0.305041, 0.050813, 0.254502, 0.389415, 0.480407, 0.537903, 0.566692, 0.568528, 0.542873, 0.486609, 0.392438, 0.243714, -0.00638205, -0.626309, -0.367357, -0.0184136, 0.119404, 0.172992, 0.171846, 0.124775, 0.0314329, -0.116698, -0.342604, -0.715674, -1.87356, -1.18355, -1.3324, -1.30782, -0.733118, -0.503491, -0.413108, -0.445614, -0.701975, -0.974846, -0.262879, 0.0369595, 0.225787, 0.353552, 0.437695, 0.485274, 0.498069, 0.473413, 0.402141, 0.259685, -0.041095, -0.920434, 0.0675066, 0.326456, 0.457087, 0.518809, 0.526446, 0.477741, 0.349984, 0.0469308, -0.556459, 0.257975, 0.523565, 0.672442, 0.760292, 0.80463, 0.811071, 0.778399, 0.697435, 0.540973, 0.203073, -0.299482, 0.421833, 0.666312, 0.800459, 0.876738, 0.91221, 0.912962, 0.879313, 0.806055, 0.678189, 0.451387, -0.122041, 0.10348, 0.487526, 0.654159, 0.738247, 0.771181, 0.76322, 0.715721, 0.622483, 0.463793, 0.173589, -1.05372, 0.0646297, 0.330393, 0.442413, 0.478661, 0.457571, 0.377924, 0.217305, -0.120147, -0.772101, 0.00737007, 0.226323, 0.320415, 0.340938, 0.295797, 0.166021, -0.144996, -0.713986, 0.0675963, 0.32451, 0.466736, 0.549717, 0.591902, 0.60074, 0.578548, 0.52386, 0.430494, 0.282809, 0.0357447, -0.563457, -0.343711, 0.0175537, 0.163497, 0.2272, 0.24004, 0.212682, 0.147374, 0.039809, -0.124974, -0.394494, -1.07615, -0.717701, -0.377144, -0.215909, -0.111074, -0.0271089, 0.0513622, 0.130248, 0.209691, 0.286775, 0.357628, 0.418597, 0.466582, 0.498852, 0.512596, 0.504232, 0.468207, 0.394339, 0.259914, -0.00688523, -1.72174, 0.00638717, 0.310287, 0.475034, 0.575544, 0.63391, 0.658348, 0.650839, 0.608537, 0.521684, 0.363777, 0.0362165, -0.5814, 0.214656, 0.468796, 0.610273, 0.694999, 0.741313, 0.756522, 0.742848, 0.698823, 0.618436, 0.486709, 0.261447, -0.276482, -0.142027, 0.261937, 0.432582, 0.51806, 0.550957, 0.540951, 0.487327, 0.378324, 0.175262, -0.320308, -0.241865, 0.193427, 0.379679, 0.477998, 0.522937, 0.525168, 0.48512, 0.39337, 0.218996, -0.172297, -0.407865, 0.159416, 0.379462, 0.500899, 0.568238, 0.596686, 0.591834, 0.554148, 0.479362, 0.355479, 0.149966, -0.272202, -0.545095, -0.0120837, 0.167788, 0.242443, 0.253225, 0.209662, 0.105829, -0.0870405, -0.493344, -0.858307, -0.311345, -0.174495, -0.191579, -0.398777, -1.33721, -0.188935, 0.125535, 0.306012, 0.417625, 0.480111, 0.499314, 0.472476, 0.384925, 0.187606, -0.448274, 0.0193243, 0.407524, 0.60707, 0.730943, 0.808419, 0.850611, 0.861638, 0.841407, 0.785538, 0.682264, 0.50001, 0.109864, -0.182809, 0.398809, 0.612583, 0.724017, 0.777912, 0.788012, 0.756708, 0.677508, 0.527959, 0.225891, -0.811029, 0.294083, 0.555555, 0.688964, 0.758463, 0.783254, 0.769388, 0.715711, 0.612967, 0.433903, 0.0735443, -0.474041, 0.228758, 0.440464, 0.535119, 0.562474, 0.533615, 0.441273, 0.247598, -0.26053, -0.118197, 0.305692, 0.490872, 0.588089, 0.630551, 0.628265, 0.580697, 0.476083, 0.275113, -0.22855, -0.118296, 0.310355, 0.496083, 0.594642, 0.640183, 0.643606, 0.606015, 0.519624, 0.358738, 0.0229391, -0.556875, 0.20493, 0.450389, 0.584821, 0.663475, 0.705027, 0.717507, 0.704311, 0.665944, 0.600137, 0.500447, 0.351332, 0.10816, -0.480053, -0.248004, 0.137165, 0.31494, 0.420297, 0.486509, 0.526797, 0.547353, 0.550976, 0.538308, 0.508136, 0.457051, 0.378117, 0.256855, 0.0564421, -0.383642, -0.47774, 0.0236294, 0.232235, 0.349543, 0.414618, 0.439973, 0.42828, 0.375069, 0.264805, 0.0489812, -0.579769, -0.18041, 0.196336, 0.378298, 0.482345, 0.537434, 0.553299, 0.531367, 0.465986, 0.339039, 0.0912876, -0.808044, -0.0176011, 0.304712, 0.465713, 0.555859, 0.599895, 0.606387, 0.576479, 0.504763, 0.374246, 0.130764, -0.633586, -0.0452115, 0.293324, 0.460152, 0.556986, 0.611957, 0.636823, 0.637509, 0.617316, 0.578198, 0.52138, 0.447708, 0.357883, 0.252558, 0.132161, -0.00408083, -0.161236, -0.358271, -0.670401, -2.07372, -0.603577, -0.282252, -0.0809483, 0.0654451, 0.175433, 0.255908, 0.309326, 0.33559, 0.332153, 0.292727, 0.202936, 0.0237186, -0.432715, -0.374638, 0.100564, 0.318135, 0.449171, 0.530815, 0.576732, 0.592575, 0.579941, 0.537285, 0.4591, 0.332219, 0.121589, -0.325291, -0.482021, 0.00830817, 0.189021, 0.273093, 0.300756, 0.284719, 0.227488, 0.124115, -0.0419143, -0.321592, -1.04814, -0.671529, -0.392124, -0.318345, -0.347061, -0.467172, -0.711034, -1.24825, -1.56643, -1.76024, -0.951792, -0.433045, -0.128642, 0.0814184, 0.233195, 0.342478, 0.417124, 0.460846, 0.474464, 0.455891, 0.398704, 0.287202, 0.0763324, -0.486954, -0.229445, 0.168715, 0.352358, 0.454679, 0.507412, 0.521846, 0.501443, 0.444649, 0.344086, 0.180672, -0.103663, -0.96601, -0.352002, -0.0926117, -0.0260547, -0.0790983, -0.301121, -1.84422, -0.193077, 0.112736, 0.276324, 0.367886, 0.407063, 0.397753, 0.33211, 0.179595, -0.197993, -0.383752, 0.185628, 0.417727, 0.550312, 0.626583, 0.66068, 0.65641, 0.610562, 0.5099, 0.3137, -0.19239, -0.045961, 0.386379, 0.582945, 0.695069, 0.756941, 0.780513, 0.769303, 0.72122, 0.62698, 0.460745, 0.127382, -0.570318, 0.253888, 0.490664, 0.607446, 0.659667, 0.662182, 0.614803, 0.501997, 0.265433, -0.643814, 0.18559, 0.516848, 0.68872, 0.791034, 0.848939, 0.871836, 0.862415, 0.818515, 0.731525, 0.578151, 0.277405, -0.848915, 0.335478, 0.604275, 0.745441, 0.824723, 0.862076, 0.864509, 0.832582, 0.76083, 0.632996, 0.399042, -0.269832, 0.164714, 0.523539, 0.695127, 0.792295, 0.844052, 0.861207, 0.847488, 0.802213, 0.719899, 0.586129, 0.360082, -0.164742, -0.0777917, 0.333252, 0.502029, 0.584639, 0.615221, 0.604871, 0.555412, 0.461241, 0.304142, 0.0248685, -0.898954, -0.159014, 0.123218, 0.238351, 0.275397, 0.255938, 0.182627, 0.0439081, -0.201575, -0.771408, -0.705373, -0.387627, -0.367578, -0.616042, -0.866941, -0.178671, 0.105172, 0.276451, 0.384648, 0.446541, 0.467711, 0.446765, 0.373568, 0.216917, -0.157655, -0.376413, 0.205788, 0.440401, 0.575925, 0.657028, 0.699158, 0.708253, 0.685354, 0.627284, 0.524559, 0.352611, 0.0259042, -0.944605, 0.0660541, 0.30037, 0.405349, 0.441806, 0.425504, 0.356262, 0.218196, -0.0476202, -0.989668, -0.20315, 0.0786893, 0.188569, 0.209561, 0.152411, -0.0125358, -0.478471, -0.383044, 0.0677674, 0.26261, 0.36289, 0.401717, 0.386103, 0.307146, 0.125151, -0.388551, -0.179893, 0.248053, 0.448523, 0.564843, 0.630567, 0.657927, 0.651262, 0.610222, 0.52963, 0.395771, 0.170871, -0.319208, -0.384872, 0.0413005, 0.180402, 0.210154, 0.153246, -0.0202786, -0.545641, -0.305988, 0.106849, 0.293538, 0.391682, 0.432505, 0.424691, 0.365429, 0.236993, -0.0245916, -1.16515, -0.0968709, 0.192128, 0.324129, 0.378552, 0.373948, 0.308534, 0.157403, -0.191967, -0.599452, 0.0556071, 0.279412, 0.389815, 0.434118, 0.4242, 0.356306, 0.205777, -0.132, -0.64325, 0.0675024, 0.292508, 0.397223, 0.429469, 0.396414, 0.279128, -0.0167993, -0.615476, 0.201993, 0.469164, 0.618404, 0.705483, 0.74788, 0.751, 0.713063, 0.623386, 0.449587, 0.0501488, -0.136127, 0.411247, 0.628321, 0.747499, 0.811468, 0.834314, 0.820058, 0.766142, 0.661198, 0.471499, 0.0466163, -0.0964925, 0.418221, 0.622776, 0.733029, 0.79051, 0.809564, 0.79526, 0.747369, 0.660104, 0.517106, 0.268229, -0.4187, 0.011263, 0.358437, 0.521912, 0.614936, 0.667948, 0.693856, 0.699423, 0.688587, 0.663603, 0.625363, 0.57322, 0.504307, 0.411874, 0.280749, 0.0704769, -0.411796, -0.35031, 0.114962, 0.331985, 0.467397, 0.556866, 0.612961, 0.640336, 0.639524, 0.607544, 0.536323, 0.406339, 0.157846, -0.725964, 0.0418147, 0.364948, 0.52495, 0.612712, 0.652559, 0.652302, 0.611804, 0.523102, 0.362488, 0.0455201, -0.9521, 0.0868856, 0.319365, 0.414411, 0.428095, 0.363355, 0.178071, -0.458761, 0.0220849, 0.405604, 0.598206, 0.711758, 0.774893, 0.797564, 0.781626, 0.722239, 0.60339, 0.374822, -0.297582, 0.147652, 0.500461, 0.662686, 0.744918, 0.773917, 0.756604, 0.688823, 0.550517, 0.267791, -1.44814, 0.281734, 0.554038, 0.687007, 0.750289, 0.76282, 0.727933, 0.637367, 0.460501, 0.0616542, -0.17197, 0.37785, 0.579729, 0.677068, 0.711349, 0.692562, 0.615241, 0.450358, 0.0645406, -0.168586, 0.397236, 0.61245, 0.725051, 0.778541, 0.78626, 0.750154, 0.66275, 0.498581, 0.158812, -0.462222, 0.304316, 0.530528, 0.637038, 0.676939, 0.663964, 0.596282, 0.454165, 0.160032, -1.07816, 0.193126, 0.448773, 0.56692, 0.612169, 0.599779, 0.525936, 0.361537, -0.0332349, -0.21869, 0.326011, 0.536902, 0.645936, 0.694757, 0.695338, 0.647238, 0.536981, 0.317211, -0.318964, 0.0706981, 0.434399, 0.600319, 0.684718, 0.715357, 0.699532, 0.633229, 0.496166, 0.212159, -1.14396, 0.257731, 0.534765, 0.679048, 0.759508, 0.79727, 0.800431, 0.771191, 0.70729, 0.60041, 0.428517, 0.117375, -1.80538, 0.0779569, 0.334261, 0.456327, 0.5168, 0.538797, 0.532978, 0.505571, 0.461075, 0.403366, 0.336003, 0.26173, 0.180922, 0.0885871, -0.0315967, -0.226809, -0.742402, -0.474354, -0.0128021, 0.235866, 0.408382, 0.536873, 0.633642, 0.704372, 0.75175, 0.776677, 0.778595, 0.755266, 0.701829, 0.608189, 0.450611, 0.150839, -1.22842, 0.171899, 0.439363, 0.574163, 0.644339, 0.669737, 0.656437, 0.602958, 0.499011, 0.313586, -0.0842483, -0.360869, 0.205851, 0.412917, 0.518505, 0.567171, 0.573046, 0.539431, 0.461929, 0.324171, 0.0730511, -0.680399, -0.140301, 0.18579, 0.333624, 0.405728, 0.428165, 0.408574, 0.344931, 0.223186, -0.00408614, -0.643146, -0.253194, 0.117308, 0.296473, 0.402395, 0.465614, 0.49819, 0.505384, 0.488838, 0.447291, 0.375799, 0.262443, 0.0761592, -0.306924, -0.639711, -0.0283279, 0.20108, 0.331297, 0.409047, 0.44971, 0.45854, 0.435314, 0.374438, 0.2607, 0.0500954, -0.497408, -0.279591, 0.125323, 0.308168, 0.406384, 0.450566, 0.44881, 0.397632, 0.277941, 0.0197345, -1.77073, 0.0418719, 0.342933, 0.502043, 0.593468, 0.638514, 0.643942, 0.608835, 0.52382, 0.360631, 0.00677748, -0.425848, 0.241046, 0.469164, 0.586319, 0.640987, 0.646544, 0.602575, 0.49437, 0.268433, -0.494122, 0.126679, 0.471605, 0.642457, 0.738686, 0.786548, 0.794953, 0.764929, 0.690467, 0.552795, 0.29016, -0.707025, 0.183191, 0.484384, 0.627457, 0.698197, 0.719573, 0.698058, 0.630969, 0.503822, 0.269239, -0.370408, -0.0183848, 0.334174, 0.486359, 0.556963, 0.574785, 0.548977, 0.480125, 0.361641, 0.175357, -0.129327, -0.815722, -0.751566, -0.722634, -0.882078, -0.122645, 0.205724, 0.411599, 0.54985, 0.640619, 0.69259, 0.708623, 0.686886, 0.619055, 0.481917, 0.193533, -0.859045, 0.283385, 0.559375, 0.707395, 0.792448, 0.834465, 0.840609, 0.811762, 0.74314, 0.620281, 0.400254, -0.150623, 0.0388022, 0.437758, 0.613397, 0.707209, 0.752404, 0.761155, 0.737947, 0.682839, 0.591534, 0.45238, 0.234091, -0.190643, -0.532031, 0.0187806, 0.19791, 0.276593, 0.302057, 0.291666, 0.254077, 0.194683, 0.117456, 0.0252335, -0.0815875, -0.209135, -0.381555, -0.694522, -1.36627, -0.473789, -0.166204, 0.0302853, 0.17059, 0.271439, 0.338888, 0.374135, 0.374541, 0.331988, 0.225828, -0.0105836, -1.20225, -0.00190887, 0.324867, 0.503185, 0.613186, 0.677892, 0.705608, 0.697936, 0.650978, 0.552251, 0.366084, -0.0699777, -0.145151, 0.347968, 0.548424, 0.653354, 0.700842, 0.701825, 0.655465, 0.548254, 0.333581, -0.28275, 0.0819072, 0.45619, 0.630054, 0.723323, 0.765153, 0.764877, 0.722863, 0.63092, 0.463375, 0.122599, -0.527441, 0.253018, 0.47708, 0.579978, 0.614989, 0.595039, 0.516074, 0.351079, -0.0214126, -0.354818, 0.249033, 0.461464, 0.565333, 0.606068, 0.596145, 0.534478, 0.405203, 0.150839, -0.734818, -0.000361685, 0.298844, 0.426149, 0.469885, 0.447674, 0.350832, 0.125354, -0.736411, 0.0325518, 0.36283, 0.526789, 0.616632, 0.657717, 0.659155, 0.622975, 0.545832, 0.416819, 0.206909, -0.192195, -0.766378, -0.134198, -0.00541242, -0.0369901, -0.251172, -2.15131, -0.171445, 0.116956, 0.250578, 0.295164, 0.25632, 0.0927461, -0.534727, -0.0169259, 0.38054, 0.587742, 0.715155, 0.791296, 0.826085, 0.821228, 0.771371, 0.658769, 0.425098, -0.438777, 0.329627, 0.663734, 0.833888, 0.931975, 0.983055, 0.99588, 0.971856, 0.906164, 0.783597, 0.557954, -0.0437126, 0.253934, 0.625948, 0.789831, 0.871098, 0.899126, 0.882383, 0.819099, 0.695576, 0.466563, -0.137316, 0.142626, 0.504206, 0.652483, 0.712365, 0.710124, 0.64712, 0.502273, 0.181535, -0.475755, 0.333113, 0.57022, 0.684656, 0.730893, 0.72291, 0.658626, 0.516995, 0.212815, -0.673914, 0.305405, 0.552784, 0.669339, 0.713926, 0.699899, 0.621059, 0.440821, -0.045008, 0.076467, 0.517171, 0.712025, 0.817241, 0.86676, 0.870854, 0.829209, 0.730489, 0.536968, 0.0582741, 0.10938, 0.555591, 0.743304, 0.839515, 0.878897, 0.871115, 0.813987, 0.690874, 0.442025, -0.510692, 0.346889, 0.660483, 0.814303, 0.896027, 0.929535, 0.922453, 0.874248, 0.775734, 0.598666, 0.229818, -0.206855, 0.441402, 0.65607, 0.76087, 0.804473, 0.801056, 0.752045, 0.647661, 0.453793, 0.00870997, -0.0731362, 0.411592, 0.605968, 0.70792, 0.757427, 0.768729, 0.747378, 0.6944, 0.607001, 0.476939, 0.283775, -0.0365783, -1.14775, -0.244074, -0.00698695, 0.0720399, 0.0769747, 0.0320055, -0.0515061, -0.16256, -0.280229, -0.364487, -0.371822, -0.302711, -0.200344, -0.102943, -0.0289431, 0.0145453, 0.0240158, -0.00458581, -0.0791155, -0.217442, -0.470926, -1.16411, -0.791768, -0.500627, -0.432664, -0.503973, -0.800178, -1.17983, -0.502105, -0.274642, -0.182529, -0.195368, -0.362175, -1.54736, -0.242345, 0.124431, 0.339942, 0.485089, 0.584211, 0.647161, 0.677671, 0.675457, 0.635812, 0.545988, 0.370531, -0.0504983, -0.124932, 0.388251, 0.607869, 0.735499, 0.811387, 0.849748, 0.855987, 0.830788, 0.770561, 0.664988, 0.48687, 0.135219, -0.47053, 0.278881, 0.504646, 0.615953, 0.666859, 0.674006, 0.642076, 0.568368, 0.440408, 0.220097, -0.280818, -0.263274, 0.16408, 0.333273, 0.414604, 0.445361, 0.439772, 0.404804, 0.344885, 0.264395, 0.170062, 0.0737505, -0.00623938, -0.0508334, -0.0539629, -0.0291799, 0.000588839, 0.0144635, -0.00523958, -0.0805022, -0.262032, -0.845698, -0.451092, -0.0482064, 0.151773, 0.268426, 0.330337, 0.34558, 0.311539, 0.211692, -0.0124715, -0.899804, -0.0765318, 0.263241, 0.441287, 0.547974, 0.608516, 0.632315, 0.622066, 0.575674, 0.484818, 0.327392, 0.029488, -2.21764, -0.00152794, 0.256, 0.370602, 0.411603, 0.395098, 0.317461, 0.14948, -0.250318, -0.425847, 0.113606, 0.324905, 0.437065, 0.492651, 0.505887, 0.481135, 0.416673, 0.30348, 0.116397, -0.235034, -1.07728, -0.206499, -0.00707033, 0.0601677, 0.0498756, -0.0290931, -0.192163, -0.508027, -1.9749, -0.718719, -0.613814, -0.933021, -0.717526, -0.182733, 0.0889921, 0.26329, 0.378829, 0.450162, 0.482692, 0.476029, 0.423072, 0.302657, 0.0414818, -1.98701, 0.0921946, 0.393453, 0.557941, 0.657969, 0.715446, 0.738895, 0.730967, 0.690143, 0.609715, 0.472257, 0.22496, -0.502379, 0.00646398, 0.343621, 0.501309, 0.585445, 0.624003, 0.627618, 0.600253, 0.541995, 0.449317, 0.313233, 0.112412, -0.218248, -1.32665, -0.473424, -0.279707, -0.282032, -0.455301, -1.31503, -0.475576, -0.128223, 0.0525193, 0.15873, 0.216715, 0.236498, 0.220902, 0.167105, 0.0638572, -0.122233, -0.5407, -0.678757, -0.151606, 0.0677667, 0.196571, 0.276846, 0.323399, 0.341833, 0.332736, 0.291993, 0.207784, 0.0478328, -0.322209, -0.53979, 0.0598944, 0.313836, 0.472905, 0.581886, 0.656418, 0.702971, 0.723748, 0.718031, 0.681915, 0.606115, 0.468221, 0.195496, -1.91577, 0.20901, 0.504159, 0.660398, 0.750818, 0.79634, 0.803504, 0.771206, 0.689791, 0.530125, 0.173721, -0.18627, 0.457024, 0.695306, 0.829051, 0.906531, 0.943674, 0.94597, 0.913157, 0.839064, 0.706316, 0.461203, -0.284842, 0.260139, 0.592688, 0.746307, 0.823299, 0.849343, 0.831549, 0.767046, 0.639992, 0.396681, -0.370853, 0.211037, 0.538654, 0.689174, 0.762444, 0.783541, 0.758929, 0.684399, 0.540267, 0.255011, -2.20587, 0.235063, 0.504739, 0.631953, 0.688789, 0.694857, 0.654117, 0.55967, 0.385485, 0.0274024, -0.523398, 0.177948, 0.38406, 0.4699, 0.484337, 0.436391, 0.313305, 0.0550711, -0.988551, -0.0509296, 0.229277, 0.341163, 0.361895, 0.297458, 0.101287, -0.674335, 0.021493, 0.375533, 0.555606, 0.658056, 0.708115, 0.713104, 0.670506, 0.564939, 0.343542, -0.391407, 0.194413, 0.54688, 0.722825, 0.824461, 0.879019, 0.896531, 0.879969, 0.82741, 0.730727, 0.568237, 0.267924, -1.70398, 0.211062, 0.467635, 0.579945, 0.620044, 0.606283, 0.539874, 0.406927, 0.155756, -0.593439, -0.0825716, 0.226099, 0.345613, 0.375372, 0.332717, 0.202992, -0.0996668, -0.878326, 0.0216508, 0.261865, 0.370184, 0.401348, 0.363374, 0.234964, -0.0972045, -0.453114, 0.207692, 0.453827, 0.591422, 0.669359, 0.703637, 0.699406, 0.655387, 0.562786, 0.396226, 0.0621388, -0.671846, 0.165956, 0.393205, 0.495967, 0.529357, 0.506265, 0.421536, 0.243893, -0.180259, -0.294214, 0.207715, 0.4028, 0.498255, 0.532955, 0.517471, 0.449555, 0.311606, 0.0397742, -1.09543, -0.0629757, 0.208333, 0.316016, 0.335049, 0.272175, 0.0863072, -0.557757, -0.0696073, 0.308643, 0.495786, 0.602542, 0.656827, 0.667395, 0.633622, 0.5446, 0.364751, -0.0768875, -0.0982891, 0.386973, 0.594154, 0.709759, 0.772059, 0.794032, 0.779355, 0.725457, 0.621327, 0.434817, 0.0281367, -0.197022, 0.349775, 0.557014, 0.665784, 0.719594, 0.732878, 0.710051, 0.649296, 0.541061, 0.35777, -0.00773577, -0.521191, 0.166703, 0.386115, 0.496276, 0.549243, 0.561944, 0.540286, 0.48413, 0.386978, 0.22967, -0.0513096, -1.1513, -0.150841, 0.146805, 0.297904, 0.386613, 0.43747, 0.460124, 0.457934, 0.43003, 0.37068, 0.264939, 0.070262, -0.436516, -0.246976, 0.200845, 0.423853, 0.568946, 0.670183, 0.740418, 0.785241, 0.806771, 0.80482, 0.776905, 0.717151, 0.612699, 0.430708, 0.0369951, -0.211767, 0.359972, 0.581396, 0.704524, 0.773757, 0.803893, 0.799853, 0.760888, 0.680118, 0.53844, 0.274728, -0.687714, 0.15642, 0.461846, 0.608948, 0.685517, 0.715711, 0.707874, 0.663184, 0.576821, 0.43425, 0.193363, -0.362983, -0.266744, 0.100656, 0.218799, 0.228243, 0.13963, -0.114874, -1.21896, -0.00119914, 0.284026, 0.434534, 0.516269, 0.54839, 0.535669, 0.473322, 0.341475, 0.068573, -2.37037, 0.0735144, 0.355924, 0.496064, 0.566006, 0.58521, 0.557149, 0.473412, 0.302236, -0.102336, -0.25422, 0.280286, 0.495579, 0.613938, 0.677263, 0.699569, 0.685051, 0.631644, 0.529317, 0.349114, -0.0244035, -0.440588, 0.19612, 0.410602, 0.517251, 0.564617, 0.567735, 0.530175, 0.447659, 0.304143, 0.0485239, -0.665314, -0.225681, 0.0903854, 0.213046, 0.247432, 0.213023, 0.0994104, -0.158563, -1.75251, -0.163145, 0.124498, 0.264206, 0.330065, 0.340815, 0.297067, 0.181767, -0.0737981, -1.83439, -0.0505439, 0.25147, 0.411798, 0.505341, 0.554083, 0.565764, 0.541196, 0.474752, 0.349129, 0.109104, -0.674103, -0.0444326, 0.294607, 0.463766, 0.561958, 0.616079, 0.636435, 0.626775, 0.586822, 0.512153, 0.391243, 0.193825, -0.198637, -0.606006, 0.00267765, 0.198916, 0.284947, 0.30548, 0.269045, 0.163812, -0.0657574, -0.934113, -0.151594, 0.185877, 0.361504, 0.467467, 0.529681, 0.558243, 0.55683, 0.525058, 0.458149, 0.343402, 0.145845, -0.289971, -0.408432, 0.0964228, 0.302084, 0.417037, 0.48248, 0.513248, 0.515501, 0.491096, 0.438528, 0.351877, 0.216108, -0.0124081, -0.575523, -0.354954, 0.0509557, 0.24645, 0.37048, 0.457245, 0.520618, 0.567446, 0.601311, 0.623902, 0.635588, 0.635641, 0.62222, 0.592063, 0.539618, 0.454744, 0.315559, 0.0562021, -1.03199, 0.00810318, 0.329212, 0.502348, 0.610259, 0.676456, 0.709678, 0.712378, 0.682574, 0.612626, 0.48271, 0.229862, -0.782471, 0.160539, 0.479099, 0.644057, 0.740754, 0.793133, 0.809797, 0.792535, 0.73772, 0.633534, 0.446649, 0.0331682, -0.151329, 0.381132, 0.587849, 0.696982, 0.750398, 0.761417, 0.732913, 0.660115, 0.526141, 0.276468, -0.501286, 0.0820057, 0.402734, 0.545852, 0.609856, 0.618147, 0.573872, 0.46451, 0.240821, -0.445491, 0.0343802, 0.384505, 0.546287, 0.627562, 0.654289, 0.632273, 0.554691, 0.392779, 0.0200374, -0.265625, 0.330269, 0.554053, 0.674029, 0.735516, 0.752627, 0.728148, 0.656212, 0.515901, 0.232423, -1.36745, 0.258376, 0.537034, 0.680611, 0.759521, 0.794944, 0.794681, 0.760342, 0.688568, 0.568672, 0.371795, -0.0129365, -0.508746, 0.136341, 0.325404, 0.397237, 0.397426, 0.331587, 0.177543, -0.177069, -0.574211, 0.0712522, 0.293521, 0.40534, 0.455008, 0.456856, 0.412684, 0.31361, 0.128713, -0.278058, -0.516739, 0.0257321, 0.221812, 0.314628, 0.346368, 0.328572, 0.259531, 0.122162, -0.146654, -1.2688, -0.234609, 0.0538075, 0.188586, 0.252115, 0.267527, 0.24277, 0.178868, 0.0713473, -0.091936, -0.339971, -0.761379, -2.77325, -1.38387, -1.19544, -0.514723, -0.179919, 0.0385839, 0.192422, 0.302499, 0.37927, 0.428419, 0.45294, 0.453908, 0.430587, 0.379846, 0.294293, 0.156483, -0.085266, -0.781276, -0.298667, 0.0616343, 0.244629, 0.359941, 0.436815, 0.486997, 0.51579, 0.525206, 0.514926, 0.48221, 0.420678, 0.316352, 0.132892, -0.289521, -0.366876, 0.154483, 0.384554, 0.525932, 0.618792, 0.677345, 0.707192, 0.709596, 0.682336, 0.618474, 0.501079, 0.280664, -0.353764, 0.0533594, 0.431607, 0.61829, 0.729919, 0.796031, 0.827591, 0.828418, 0.797933, 0.730829, 0.613082, 0.405174, -0.0872905, -0.0289314, 0.411984, 0.601535, 0.704875, 0.757831, 0.772836, 0.753702, 0.698725, 0.599444, 0.432463, 0.11583, -0.975031, 0.149799, 0.400092, 0.522769, 0.582938, 0.600641, 0.583377, 0.532936, 0.446868, 0.31735, 0.125619, -0.181985, -0.948827, -0.600008, -0.388761, -0.443901, -0.890558, -0.615827, -0.156847, 0.0694921, 0.207295, 0.292095, 0.337949, 0.351157, 0.334006, 0.285763, 0.20217, 0.0727192, -0.129219, -0.499788, -1.22122, -0.443279, -0.243817, -0.161585, -0.137869, -0.150854, -0.186848, -0.232366, -0.271536, -0.288771, -0.276652, -0.241349, -0.198366, -0.164576, -0.155159, -0.186801, -0.289329, -0.561404, -1.15697, -0.294276, 0.0064247, 0.19133, 0.317587, 0.404359, 0.459919, 0.487886, 0.489109, 0.462034, 0.401964, 0.298359, 0.125585, -0.204655, -1.10248, -0.151133, 0.0771361, 0.174121, 0.195911, 0.150509, 0.0165315, -0.318598, -0.687898, -0.0242458, 0.220556, 0.356616, 0.432547, 0.463465, 0.452726, 0.395029, 0.270566, 0.00932175, -1.71249, 0.0256464, 0.327892, 0.489687, 0.586165, 0.639578, 0.658216, 0.64403, 0.593963, 0.497607, 0.326217, -0.0328295, -0.467842, 0.20044, 0.433862, 0.561408, 0.633376, 0.666411, 0.666618, 0.63463, 0.566031, 0.44792, 0.243741, -0.216271, -0.264757, 0.208747, 0.404576, 0.511645, 0.568935, 0.590433, 0.581675, 0.543847, 0.474568, 0.366587, 0.202286, -0.0699974, -0.79053, -0.375319, -0.0653882, 0.0542004, 0.0921466, 0.0738845, 0.00469567, -0.121905, -0.329593, -0.698283, -2.61317, -0.98056, -1.06801, -1.24078, -0.5117, -0.199937, -0.00991958, 0.109594, 0.175013, 0.188927, 0.141152, -0.00782001, -0.47976, -0.259822, 0.208133, 0.445951, 0.600921, 0.707673, 0.779447, 0.821976, 0.837175, 0.824067, 0.778222, 0.688947, 0.529164, 0.200154, -0.425943, 0.370899, 0.620871, 0.756201, 0.832228, 0.866213, 0.863625, 0.823258, 0.736307, 0.577229, 0.251041, -0.436814, 0.39445, 0.640858, 0.769311, 0.83647, 0.859582, 0.843527, 0.785652, 0.673442, 0.468669, -0.0170872, 0.0143438, 0.454447, 0.633703, 0.720643, 0.749322, 0.72785, 0.650336, 0.488585, 0.117388, -0.176649, 0.42408, 0.649723, 0.772163, 0.837461, 0.860648, 0.84621, 0.792034, 0.687627, 0.502232, 0.107992, -0.195883, 0.380439, 0.585821, 0.687652, 0.73092, 0.729445, 0.685724, 0.593554, 0.431587, 0.124239, -1.44983, 0.0983837, 0.343226, 0.447074, 0.476855, 0.447783, 0.354986, 0.165482, -0.291373, -0.326458, 0.13527, 0.315363, 0.398038, 0.419086, 0.386841, 0.29481, 0.112181, -0.305375, -0.484161, 0.0318384, 0.221243, 0.307725, 0.331307, 0.302169, 0.216419, 0.0511747, -0.284939, -1.05061, -0.208365, 0.00248521, 0.0818079, 0.0826881, 0.0125068, -0.149102, -0.501119, -1.12337, -0.416168, -0.267593, -0.305342, -0.644368, -0.594175, -0.0426702, 0.219459, 0.383069, 0.488447, 0.549992, 0.57261, 0.555152, 0.489109, 0.349543, 0.0440129, -0.67827, 0.201362, 0.461699, 0.600652, 0.675802, 0.704028, 0.688894, 0.624377, 0.48787, 0.194394, -0.708409, 0.312543, 0.585005, 0.733481, 0.820737, 0.866636, 0.87894, 0.859897, 0.807694, 0.715304, 0.564779, 0.301638, -0.472938, 0.0744401, 0.389088, 0.527197, 0.590432, 0.605474, 0.581792, 0.521771, 0.422958, 0.277101, 0.0644919, -0.270624, -1.08588, -0.761211, -0.630584, -0.867058, -1.21192, -0.549754, -0.350125, -0.315744, -0.469774, -1.87869, -0.244909, 0.111264, 0.3259, 0.471475, 0.571259, 0.63525, 0.66784, 0.670024, 0.639678, 0.570201, 0.445497, 0.219305, -0.37448, -0.0971185, 0.277134, 0.440064, 0.519261, 0.544048, 0.522297, 0.450752, 0.311473, 0.040699, -1.08639, -0.0526333, 0.23031, 0.356655, 0.406756, 0.400081, 0.336864, 0.198489, -0.0912373, -1.38325, -0.0567647, 0.205556, 0.332347, 0.391036, 0.401304, 0.369036, 0.292501, 0.161301, -0.054421, -0.471151, -0.950409, -0.381917, -0.265164, -0.303772, -0.515324, -1.68221, -0.540444, -0.261929, -0.162911, -0.183434, -0.372685, -2.32488, -0.225146, 0.108345, 0.296786, 0.412725, 0.477254, 0.496443, 0.467175, 0.373357, 0.159655, -0.623848, 0.0636041, 0.41703, 0.599771, 0.708348, 0.76863, 0.789569, 0.772492, 0.71215, 0.591981, 0.360069, -0.34356, 0.154031, 0.499038, 0.658379, 0.737906, 0.76302, 0.739059, 0.658441, 0.490169, 0.0921439, -0.0835066, 0.459681, 0.673969, 0.788601, 0.845477, 0.857623, 0.827087, 0.746992, 0.593997, 0.281747, -0.613481, 0.364406, 0.610263, 0.727318, 0.774938, 0.76767, 0.702501, 0.554271, 0.211703, -0.203203, 0.469648, 0.707403, 0.835551, 0.903647, 0.92713, 0.909762, 0.847183, 0.722449, 0.479203, -0.345443, 0.337502, 0.665061, 0.8244, 0.910755, 0.949596, 0.949507, 0.911231, 0.828256, 0.680326, 0.398413, -1.1066, 0.356995, 0.640109, 0.780984, 0.856479, 0.889473, 0.88899, 0.858279, 0.796932, 0.700732, 0.559019, 0.344674, -0.0408774, -0.731661, 0.00640107, 0.19173, 0.259878, 0.267406, 0.234379, 0.172699, 0.095166, 0.0212812, -0.022187, -0.0143581, 0.0412078, 0.120797, 0.20062, 0.265863, 0.308722, 0.324424, 0.3081, 0.251685, 0.137729, -0.0838375, -0.730862, -0.316881, 0.0462268, 0.214709, 0.302705, 0.337925, 0.328275, 0.271487, 0.152628, -0.0790269, -0.779114, -0.28954, 0.0562546, 0.217151, 0.300773, 0.334605, 0.327781, 0.281125, 0.187832, 0.0265024, -0.277785, -1.93036, -0.296447, -0.0396987, 0.0811892, 0.136694, 0.147826, 0.12193, 0.0594432, -0.0462085, -0.215946, -0.525188, -1.94202, -0.509638, -0.22902, -0.0680628, 0.0445771, 0.131929, 0.203801, 0.264438, 0.315182, 0.355726, 0.384787, 0.400417, 0.400021, 0.38005
};


 #endif /* MULTIPATH_V30_M10_H_ */
 
