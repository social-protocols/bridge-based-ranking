using StatsBase
using CSV
using DataFrames
using SparseArrays

using SQLite

function loadCommunityNotesRatingsMatrix(; table = "sampleDataSet2", coreOnly = false)
	db = SQLite.DB("~/community-notes-data/community-notes-data.sqlite")

	coreOnlyQuery = coreOnly ? " and modelingPopulation = 'CORE'" : "" 
	query = "SELECT sampleDataSet.*, currentStatus, enrollmentState FROM $(table) sampleDataSet join currentStatus using (noteId) join userEnrollment on (participantId = raterParticipantId) 
	where version = 2$(coreOnlyQuery)"

	d = DataFrame(DBInterface.execute(db, query))


	unique_users = unique(d.raterParticipantId)
	unique_items = unique(d.noteId)
	user_indices = Dict(pid => idx for (idx, pid) in enumerate(unique_users))
	item_indices = Dict(iid => idx for (idx, iid) in enumerate(unique_items))

	ratings_matrix = SparseArrays.spzeros(size(unique_users)[1], size(unique_items)[1])
	mask_matrix = SparseArrays.spzeros(size(unique_users)[1], size(unique_items)[1])

	h = Dict("HELPFUL" => 1, "NOT_HELPFUL" => -1, "SOMEWHAT_HELPFUL" => .5)
	r = 1


	colorIndex = Dict("CURRENTLY_RATED_HELPFUL" => :green, "CURRENTLY_RATED_NOT_HELPFUL" => :red, "NEEDS_MORE_RATINGS" => :gray)
	enrollmentStateIndex = Dict("earnedIn" => :green, "newUser" => :gray, "atRisk" => :hellow, "earnedOutAcknowledged" => :orange, "earnedOutAcknowledged" => :earnedOutNoAcknowledge)
	
	userColors = Vector{Symbol}(undef, length(unique_users))
	itemColors = Vector{Symbol}(undef, length(unique_items))

	
	userIds = Vector{String}(undef, length(unique_users))
	itemIds = Vector{String}(undef, length(unique_items))


	@showprogress for row in eachrow(d)
	    user_idx = user_indices[row.raterParticipantId]
	    item_idx = item_indices[row.noteId]
	    # print(user_idx, ", ", item_idx, ",", row.helpfulnessLevel,"\n")

	    if (row.helpfulnessLevel === missing)
	    	rating = row.helpful - row.notHelpful
	    	# println("Missing", row.helpful, row.notHelpful, row.helpfulnessLevel, rating)
	    	# r = row
	    elseif (row.helpfulnessLevel === "") 
	    	println("Row: ", row)
	    else
	    	rating = h[row.helpfulnessLevel]
	    end
	    
	    ratings_matrix[user_idx, item_idx] = rating

	    itemColors[item_idx] = colorIndex[row.currentStatus]
	    userColors[user_idx] = colorIndex[row.currentStatus]

	    itemIds[item_idx] = row.noteId
	    userIds[user_idx] = row.raterParticipantId
	end

	return (ratings_matrix, userColors, itemColors, userIds, itemIds)
end

