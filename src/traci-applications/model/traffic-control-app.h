#ifndef TRAFFIC_INFO_SERVER_H
#define TRAFFIC_INFO_SERVER_H

#include "ns3/opengym-module.h"
#include "ns3/traci-client.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/address.h"
#include "ns3/ipv4-address.h"
#include "ns3/traced-callback.h"


namespace ns3 {

	std::vector<std::string> split(const std::string& input, const std::string& regex);

	class Socket;
	class Packet;

		/**
	 * \ingroup applications 
	 * \defgroup TrafficInfo TrafficInfo
	 */
	
	
	/**
		 * \brief The RL environment for the RSU to contril speeds 
		 * an object of this class is used in the RsuControl class 
		 */
	class RsuEnv : public OpenGymEnv {
	public:
		RsuEnv();
		virtual ~RsuEnv();
		static TypeId GetTypeId(void);
		virtual void DoDispose();

		// GYM specific methods
		/**
		 * \brief called on Notify. Used to set observation space params for the agent. The observation space is responsible for values processed by agent.
		 * \return a pointer to OpenGymSpace object.
		 */
		Ptr<OpenGymSpace> GetObservationSpace();

		/**
		 * \brief called on Notify. Used to set Action space params for the agent. THe Action space consists of agent-specific actions that depend on observations.
		 * \return a pointer to OpenGymSpace object.
		 */
		Ptr<OpenGymSpace> GetActionSpace();
		
		/**
		 * \brief called on Notify. Used to send new observations (new speeds and headways) to the agent.
		 * \return a pointer to OpenGymDataContainer object that contains observation values.
		 */
		Ptr<OpenGymDataContainer> GetObservation();
		
		/**
		 * \brief called on Notify. Calculates the reward based on actions performed by the agent.
		 * \return the float value of the reward.
		 */
		float GetReward();
		bool GetGameOver();
		std::string GetExtraInfo();
		
		/**
		 * \brief called on Notify. Collects new agent actions and saves them.
		 * \return
		 */
		bool ExecuteActions(Ptr<OpenGymDataContainer> action);
		
		/**
		 * \brief Called from RSU class to get new speed values for vehicles
		 * \return vector of speeds
		 */
		std::vector<float> ExportNewSpeeds();
		
		/**
		 * \brief Called from RSU class to supply new observation data (current headways and velocities)
		 */
		void ImportSpeedsAndHeadWays(std::vector<double> RSU_headways,std::vector<double> RSU_speeds);
		
		
		uint32_t m_vehicles;					//!< Number of vehicles
		double m_alpha;							//!< Constant for reward
		double m_beta;							//!< Constant for reward
		double max_headway_time;				//!< maximum headway in seconds
		double max_velocity_value;				//!< Maximum velocity in m/s
		// Look into this
		double desired_velocity_value;			//!< Desired velocity value which is less than but close to max
		double old_reward;
		double current_reward;
		float epsilon_threshold;
		uint32_t current_step;
		uint32_t horizon;
		
		std::vector<double> actual_speeds;		//!< Vecor of current vehicle speeds in the environment
		std::vector<double> actual_headways;	//!< Vecor of current vehicle headways in the environment
		std::vector<float> new_speeds;		//!< Vecor of new vehicle speeds to be transmitted
	};

	/**
	 * \ingroup TrafficInfo
	 * \brief A Traffic Info server
	 *
	 * Traffic information is broadcasted
	 */
	class RsuSpeedControl : public Application {
	public:
		/**
		 * \brief Get the type ID.
		 * \return the object TypeId
		 */
		static TypeId GetTypeId(void);
		RsuSpeedControl();
		virtual ~RsuSpeedControl();
		Ptr<RsuEnv> GetEnv();


	protected:
		virtual void DoDispose(void);

	private:

		virtual void StartApplication(void);
		virtual void StopApplication(void);

		/**
		 * \brief Schedule the next packet transmission
		 * \param dt time interval between packets.
		 */
		void ScheduleTransmit(Time dt);

		/**
		 * \brief Send a packet
		 */
		void Send(void);
		void ChangeSpeed(void);
		void HandleRead(Ptr<Socket> socket);
		

		uint16_t m_port; //!< Port on which traffic information is sent
		Time m_interval; //!< Packet inter-send time
		uint32_t m_count; //!< Maximum number of packets the application will send
		Ptr<Socket> tx_socket; //!< IPv4 Socket
		Ptr<Socket> rx_socket; //!< IPv4 Socket
		EventId m_sendEvent; //!< Event to send the next packet
		Ptr<TraciClient> m_client;
		std::map<std::string, std::pair<double, double>> m_vehicles_data;

		/// Callbacks for tracing the packet Tx events
		TracedCallback<Ptr<const Packet> > m_txTrace;
		
		// GymEnv
		Ptr<RsuEnv> m_rsuGymEnv; //!< Gym environment object

	};

	class VehicleSpeedControl : public Application {
	public:
		/**
		 * \brief Get the type ID.
		 * \return the object TypeId
		 */
		static TypeId GetTypeId(void);

		VehicleSpeedControl();

		virtual ~VehicleSpeedControl();

		void StopApplicationNow();

	protected:
		virtual void DoDispose(void);

	private:

		virtual void StartApplication(void);
		virtual void StopApplication(void);

		/**
		 * \brief Handle a packet reception.
		 *
		 * This function is called by lower layers.
		 *
		 * \param socket the socket the packet was received to.
		 */
		void HandleRead(Ptr<Socket> socket);
		void Send(void);
		void ScheduleTransmit(Time dt);

		Ptr<Socket> rx_socket; //!< Socket
		uint16_t m_port; //!< Port on which client will listen for traffic information
		EventId m_sendEvent; //!< Event to send the next packet
		Time m_interval; //!< Packet inter-send time
		Ptr<Socket> tx_socket; //!< Socket
		Ptr<TraciClient> m_client;
		double last_velocity;
		double last_headway;
	};


} // namespace ns3

#endif /* TRAFFIC_INFO_SERVER_H */

