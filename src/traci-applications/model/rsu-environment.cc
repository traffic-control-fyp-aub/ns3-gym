/**
    EECE 502 
    rsu-environment.cc
    Purpose: Defines the agent's OpenAI Gym environment
    @author Rayyan Nasr 
    @author Jihad Eddine Al-Khurfan
    @version 1.0 4/1/20
*/

#include "ns3/log.h"
#include "rsu-environment.h"
#include <cmath>


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("RsuRnvironment");
NS_OBJECT_ENSURE_REGISTERED (RsuEnv);

/**
    The constructor of the Gym environment. It is used to define the environment
    parameters (i.e. number of vehicles) that we will handle in the Observation Space
    and the number of outputs in the Action Space. Once this consturctor is called,
    the ZMQ bridge is set up between ns-3 and the RL agent.
*/
RsuEnv::RsuEnv ()
{
  NS_LOG_FUNCTION (this);
  // Opening interface with simulation script
  this->SetOpenGymInterface (OpenGymInterface::Get ());
  // Setting default values for params
  m_vehicles = 0;
  m_max_vehicles = 25;
  m_alpha = 0.9;
  m_beta = 0.99;
  max_headway_time = 2.0;
  max_velocity_value = 50; // was 50
  desired_velocity_value = 45; // was 47
  old_reward = 0.0;
  current_reward = 0.0;
  current_step = 1;
  horizon = 128;
  epsilon_threshold = 1e-4;
  max_delta = 6.0;

  NS_LOG_INFO ("Set Up Interface : " << OpenGymInterface::Get () << "\n");
}

RsuEnv::~RsuEnv ()
{
  NS_LOG_FUNCTION (this);
}

/**
    Gets type id of the RSU environment.

    @return type id.
*/
TypeId
RsuEnv::GetTypeId (void)
{
  static TypeId tid = TypeId ("RsuEnv")
                          .SetParent<OpenGymEnv> ()
                          .SetGroupName ("Applications")
                          .AddConstructor<RsuEnv> ();
  return tid;
}

void
RsuEnv::DoDispose ()
{
  NS_LOG_FUNCTION (this);
}

/**
    Sends the Observation Space definition to the RL Agent.

    @return Pointer to the Observation Space.
*/
Ptr<OpenGymSpace>
RsuEnv::GetObservationSpace ()
{
  NS_LOG_FUNCTION (this);

  // set low and high values: low is for lowest speed/headway while high is for highest
  float low = 0.0;
  float high = max_velocity_value;

  // setting observation space shape which has a size of 2*numOfVehicles since it has headways and velocities for each vehicle
  std::vector<uint32_t> shape = {
      2 * m_max_vehicles,
  };
  std::string dtype = TypeNameGet<float> ();

  // initializing observation space
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
  NS_LOG_UNCOND ("GetObservationSpace: " << space);
  return space;
}

/**
    Sends the Action Space definition to the RL Agent.

    @return Pointer to the Action Space.
*/
Ptr<OpenGymSpace>
RsuEnv::GetActionSpace ()
{
  NS_LOG_FUNCTION (this);

  // set low and high values
  float low = -max_delta;
  float high = max_delta;

  // setting action space shape which has a size of numOfVehicles since actions are respective speeds for each vehicles
  std::vector<uint32_t> shape = {
      m_max_vehicles,
  };
  std::string dtype = TypeNameGet<float> ();

  // initializing action space
  Ptr<OpenGymBoxSpace> box = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);
  NS_LOG_INFO ("GetActionSpace: " << box);
  return box;
}

/**
    Collects the state of the environment. Used when the RL Agent needs to predict
    an action since the agent needs to know the current state of the environment.

    @return Pointer to the observation shape.
*/
Ptr<OpenGymDataContainer>
RsuEnv::GetObservation ()
{
  NS_LOG_FUNCTION (this);

  // setting observation shape which has a size of 2*numOfVehicles since it has headways and velocities for each vehicle
  std::vector<uint32_t> shape = {
      2 * m_max_vehicles,
  };
  Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>> (shape);

  // send zeros first time

  // Add Current headways of vehicles reachable by RSU to the observation
  for (uint32_t i = 0; i < actual_headways.size (); ++i)
    {
      float value = static_cast<float> (actual_headways[i]);
      box->AddValue (value);
    }

  // Add Current velocities of vehicles reachable by RSU to the observation
  for (uint32_t i = 0; i < actual_speeds.size (); ++i)
    {
      float value = static_cast<float> (actual_speeds[i]);
      box->AddValue (value);
    }

  NS_LOG_UNCOND ("MyGetObservation: " << box);
  return box;
}

/**
    Computes the reward and sends it to the RL agent over the ZMQ bridge. It is used by the agent
    to know how good the previously predicted actions are in order to learn for better performance.

    @return The value of the reward.
*/
float
RsuEnv::GetReward ()
{
  NS_LOG_FUNCTION (this);

  // The following formula is used to calculate the reward for the agent:
  // reward = beta * ( max_v - sum(|desired_v - v[i]|)/N - alpha * sum(max(max_h - h[i])))

  float reward = 0.0;
  double max_headway_summation = 0.0;
  for (uint32_t i = 0; i < actual_headways.size (); i++)
    {
      max_headway_summation += fmax (max_headway_time - actual_headways[i], 0.0);
    }
  double abs_speed_diff_summation = 0.0;
  for (uint32_t i = 0; i < actual_speeds.size (); i++)
    {
      abs_speed_diff_summation += abs (desired_velocity_value - actual_speeds[i]);
    }
  reward = max_velocity_value - (abs_speed_diff_summation / m_vehicles) -
           (max_headway_summation * m_alpha);

  current_reward = reward;
  if (current_step % horizon == 0)
    {
      old_reward = current_reward;
    }

  reward *= m_beta;

  NS_LOG_UNCOND ("MyGetReward: " << reward);
  return reward;
}

/**
    Prematurely terminates the episode of training in the case where the RL agent reaches an end
    state in training. In this case, the cause of reaching an end state could either be an accident
    or an invariable state of the reward after many time steps.

    @return Boolean value of isGameOver, which is initially set to false.
*/
bool
RsuEnv::GetGameOver ()
{
  NS_LOG_FUNCTION (this);
  bool isGameOver = false;
  //	isGameOver = pow(abs(old_reward - current_reward), 2) < epsilon_threshold;
  NS_LOG_UNCOND ("MyGetGameOver: " << isGameOver);
  return isGameOver;
}

std::string
RsuEnv::GetExtraInfo ()
{
  NS_LOG_FUNCTION (this);
  std::string myInfo = "info";
  NS_LOG_UNCOND ("MyGetExtraInfo: " << myInfo);
  return myInfo;
}

/**
    Receives the predicted actions from the RL agent and physically implements these actions on the
    vehicles in SUMO.

    @param action The predicted action from the RL agent.
    @return Boolean value true.
*/
bool
RsuEnv::ExecuteActions (Ptr<OpenGymDataContainer> action)
{
  NS_LOG_FUNCTION (this);

  // get the latest actions performed by the agent
  Ptr<OpenGymBoxContainer<float>> box = DynamicCast<OpenGymBoxContainer<float>> (action);

  // get new actions data (velocities)
  new_speeds = box->GetData ();

  // make sure all values are in the range [+]
  //	for (uint32_t i = 0; i < new_speeds.size(); i++) {
  //		new_speeds[i] = new_speeds[i] < 0 ? - fmod(abs(new_speeds[i]), max_delta) : fmod(abs(new_speeds[i]), max_delta);
  //	}

  current_step++;
  NS_LOG_UNCOND ("MyExecuteActions: " << action);
  return true;
}

/**
    Exports the speeds of the vehicles from SUMO. This function handles any instance where there is a
    mismatch between the number of vehicles on the traffic scenario and the size of the Observation
    and Action Spaces.

    @return Speeds of the vehicles.
*/
std::vector<float>
RsuEnv::ExportNewSpeeds ()
{
  NS_LOG_FUNCTION (this);
  std::vector<float> new_speeds_no_paddings;
  // Remove unecessary paddings from new speeds
  for (uint32_t i = 0; i < m_vehicles; i++)
    {
      new_speeds_no_paddings.push_back (new_speeds[i]);
    }
  NS_LOG_INFO ("###################################################################################"
               "########################\n");
  return new_speeds_no_paddings;
}

/**
    Imports the speeds and headways from SUMO.

    @param RSU_headways The headways of the vehicles in SUMO.
    @param RSU_speeds The speeds of the vehicles in SUMO.
*/
void
RsuEnv::ImportSpeedsAndHeadWays (std::vector<double> RSU_headways, std::vector<double> RSU_speeds)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_INFO ("###################################################################################"
               "########################\n");

  // remove old headway and speed values
  actual_headways.clear ();
  actual_speeds.clear ();

  // get new speed and headway values from RSU
  actual_headways = RSU_headways;
  // pad zeros to eliminate missmatch with headways
  for (uint32_t i = actual_headways.size (); i < m_max_vehicles; i++)
    {
      actual_headways.push_back (0.0);
    }
  actual_speeds = RSU_speeds;
  // pad zeros to eliminate missmatch with speeds
  for (uint32_t i = actual_speeds.size (); i < m_max_vehicles; i++)
    {
      actual_speeds.push_back (0.0);
    }
  m_vehicles = actual_speeds.size ();
  Notify ();
}

/**
    Returns the size of the Action Space.

    @return The maximum number of vehicles in the Action Space.
*/
uint32_t
RsuEnv::GetActionSpaceSize ()
{
  return m_max_vehicles;
}

} // namespace ns3