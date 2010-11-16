package de.lmu.ifi.dbs.elki.database.query;

import java.util.List;

import de.lmu.ifi.dbs.elki.data.DatabaseObject;
import de.lmu.ifi.dbs.elki.database.Database;
import de.lmu.ifi.dbs.elki.database.DistanceResultPair;
import de.lmu.ifi.dbs.elki.database.ids.DBID;
import de.lmu.ifi.dbs.elki.database.query.distance.DistanceQuery;
import de.lmu.ifi.dbs.elki.distance.distancefunction.DistanceFunction;
import de.lmu.ifi.dbs.elki.distance.distancevalue.Distance;

/**
 * Utility classes for Database Query handling.
 * 
 * @author Erich Schubert
 */
public final class DatabaseQueryUtil {
  /**
   * Execute a single kNN query by Object DBID
   * 
   * @param <O> Object type
   * @param <D> Distance type
   * @param database Database to query
   * @param distanceFunction Distance function to use
   * @param k Value of k
   * @param id DBID to query
   */
  public static <O extends DatabaseObject, D extends Distance<D>> List<DistanceResultPair<D>> singleKNNQueryByID(Database<O> database, DistanceFunction<? super O, D> distanceFunction, int k, DBID id) {
    return database.getKNNQuery(distanceFunction, k).getForDBID(id);
  }

  /**
   * Execute a single kNN query by Object DBID
   * 
   * @param <O> Object type
   * @param <D> Distance type
   * @param database Database to query
   * @param distanceQuery Distance query to use
   * @param k Value of k
   * @param id DBID to query
   */
  public static <O extends DatabaseObject, D extends Distance<D>> List<DistanceResultPair<D>> singleKNNQueryByID(Database<O> database, DistanceQuery<O, D> distanceQuery, int k, DBID id) {
    return database.getKNNQuery(distanceQuery, k).getForDBID(id);
  }

  /**
   * Execute a single kNN query by Object
   * 
   * @param <O> Object type
   * @param <D> Distance type
   * @param database Database to query
   * @param distanceFunction Distance function to use
   * @param k Value of k
   * @param obj Query object
   */
  public static <O extends DatabaseObject, D extends Distance<D>> List<DistanceResultPair<D>> singleKNNQueryByObject(Database<O> database, DistanceFunction<? super O, D> distanceFunction, int k, O obj) {
    return database.getKNNQuery(distanceFunction, k).getForObject(obj);
  }

  /**
   * Execute a single kNN query by Object
   * 
   * @param <O> Object type
   * @param <D> Distance type
   * @param database Database to query
   * @param distanceQuery Distance query to use
   * @param k Value of k
   * @param obj Query object
   * @return
   */
  public static <O extends DatabaseObject, D extends Distance<D>> List<DistanceResultPair<D>> singleKNNQueryByObject(Database<O> database, DistanceQuery<O, D> distanceQuery, int k, O obj) {
    return database.getKNNQuery(distanceQuery, k).getForObject(obj);
  }
}