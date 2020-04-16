#ifndef RSU_ENVIRONMENT_H
#define RSU_ENVIRONMENT_H

#include "ns3/opengym-module.h"

namespace ns3 {
/**
		 * \brief The RL environment for the RSU to control speeds 
		 * an object of this class is used in the RsuControl class 
		 */
class RsuSpeedControlEnv : public OpenGymEnv
{
public:
  RsuSpeedControlEnv ();
  virtual ~RsuSpeedControlEnv ();
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  // GYM specific methods
  /**
		 * \brief called on Notify. Used to set observation space params for the agent. The observation space is responsible for values processed by agent.
		 * \return a pointer to OpenGymSpace object.
		 */
  Ptr<OpenGymSpace> GetObservationSpace ();

  /**
		 * \brief called on Notify. Used to set Action space params for the agent. THe Action space consists of agent-specific actions that depend on observations.
		 * \return a pointer to OpenGymSpace object.
		 */
  Ptr<OpenGymSpace> GetActionSpace ();

  /**
		 * \brief called on Notify. Used to send new observations (new speeds and headways) to the agent.
		 * \return a pointer to OpenGymDataContainer object that contains observation values.
		 */
  Ptr<OpenGymDataContainer> GetObservation ();

  /**
		 * \brief called on Notify. Calculates the reward based on actions performed by the agent.
		 * \return the float value of the reward.
		 */
  float GetReward ();
  bool GetGameOver ();
  std::string GetExtraInfo ();

  /**
		 * \brief called on Notify. Collects new agent actions and saves them.
		 * \return
		 */
  bool ExecuteActions (Ptr<OpenGymDataContainer> action);

  /**
		 * \brief Called from RSU class to get new speed values for vehicles
		 * \return vector of speeds
		 */
  std::vector<float> ExportNewSpeeds ();

  /**
		 * \brief Called from RSU class to supply new observation data (current headways and velocities)
		 */
  void ImportSpeedsAndHeadWays (std::vector<double> RSU_headways, std::vector<double> RSU_speeds);

  /**
		 * \brief Returns size of actionspace
		 */
  uint32_t GetActionSpaceSize ();

private:
  uint32_t m_vehicles; //!< Number of vehicles
  uint32_t m_max_vehicles; //!< Number of vehicles
  double m_alpha; //!< Constant for reward
  double m_beta; //!< Constant for reward
  double max_headway_time; //!< maximum headway in seconds
  double max_velocity_value; //!< Maximum velocity in m/s
  // Look into this
  double desired_velocity_value; //!< Desired velocity value which is less than but close to max
  double old_reward;
  double current_reward;
  float max_delta;
  float epsilon_threshold;
  uint32_t current_step;
  uint32_t horizon;

  std::vector<double> actual_speeds; //!< Vecor of current vehicle speeds in the environment
  std::vector<double> actual_headways; //!< Vecor of current vehicle headways in the environment
  std::vector<float> new_speeds; //!< Vecor of new vehicle speeds to be transmitted
};

/**
		 * \brief The RL environment for the RSU to control lane changing behavior 
		 * an object of this class is used in the RsuControl class 
		 */
class RsuLaneControlEnv : public OpenGymEnv
{
public:
  RsuLaneControlEnv ();
  virtual ~RsuLaneControlEnv ();
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  // GYM specific methods
  /**
		 * \brief called on Notify. Used to set observation space params for the agent. The observation space is responsible for values processed by agent.
		 * \return a pointer to OpenGymSpace object.
		 */
  Ptr<OpenGymSpace> GetObservationSpace ();

  /**
		 * \brief called on Notify. Used to set Action space params for the agent. THe Action space consists of agent-specific actions that depend on observations.
		 * \return a pointer to OpenGymSpace object.
		 */
  Ptr<OpenGymSpace> GetActionSpace ();

  /**
		 * \brief called on Notify. Used to send new observations to the agent.
		 * \return a pointer to OpenGymDataContainer object that contains observation values.
		 */
  Ptr<OpenGymDataContainer> GetObservation ();

  /**
		 * \brief called on Notify. Calculates the reward based on actions performed by the agent.
		 * \return the float value of the reward.
		 */
  float GetReward ();
  bool GetGameOver ();
  std::string GetExtraInfo ();

  /**
		 * \brief called on Notify. Collects new agent actions and saves them.
		 * \return
		 */
  bool ExecuteActions (Ptr<OpenGymDataContainer> action);

  // /**
	// 	 * \brief Called from RSU class to get new speed values for vehicles
	// 	 * \return vector of speeds
	// 	 */
  // std::vector<float> ExportNewSpeeds ();

  // /**
	// 	 * \brief Called from RSU class to supply new observation data (current headways and velocities)
	// 	 */
  // void ImportSpeedsAndHeadWays (std::vector<double> RSU_headways, std::vector<double> RSU_speeds);

  /**
		 * \brief Returns size of actionspace
		 */
  uint32_t GetActionSpaceSize ();

private:
  uint32_t m_vehicles; //!< Number of vehicles
  uint32_t m_max_vehicles; //!< Number of vehicles
  double m_alpha; //!< Constant for reward
  double m_beta; //!< Constant for reward
  // double max_headway_time; //!< maximum headway in seconds
  // double max_velocity_value; //!< Maximum velocity in m/s
  // // Look into this
  // double desired_velocity_value; //!< Desired velocity value which is less than but close to max
  double old_reward;
  double current_reward;
  float max_delta;
  float epsilon_threshold;
  uint32_t current_step;
  uint32_t horizon;

  // std::vector<double> actual_speeds; //!< Vecor of current vehicle speeds in the environment
  // std::vector<double> actual_headways; //!< Vecor of current vehicle headways in the environment
  // std::vector<float> new_speeds; //!< Vecor of new vehicle speeds to be transmitted
};

} // namespace ns3

#endif